"""
train_ct_model.py
==================
Fine-tunes the ResNet18 backbone used by `predict_ct_risk()` in
ml_engine.py on a labeled CT image dataset, and saves the weights so
ml_engine.py picks them up automatically.

--------------------------------------------------------------------------
IMPORTANT -- ABOUT LABELS
--------------------------------------------------------------------------
This script does NOT assume any specific dataset. You point it at:
  1. An image folder (e.g. tiff_images/), and
  2. A CSV/Excel file with a filename column + a label column.

If your CSV's label column isn't actually a stroke/hemorrhage label
(e.g. the Kaggle "CT Medical Images" dataset only has an Age/Contrast
column, no stroke annotation), the resulting model will faithfully learn
THAT column -- not stroke risk. Point --label-col at a real
stroke/hemorrhage label once you have a dataset that provides one
(e.g. Kaggle "Head CT - Hemorrhage", RSNA Intracranial Hemorrhage
Detection, or CQ500). Until then, only use this to sanity-check the
pipeline, not to produce a clinically meaningful model.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python train_ct_model.py \
        --image-dir /path/to/tiff_images \
        --labels /path/to/overview.csv \
        --filename-col tiff_name \
        --label-col stroke \
        --epochs 10

--labels accepts .csv or .xlsx.
--label-col values are coerced to 0/1: numeric 0/1, True/False, or any
two-valued category (the first unique value seen -> 0, the other -> 1;
printed at runtime so you can verify the mapping is what you expect).

Output:
    ct_resnet.pt   -- PyTorch state_dict, auto-loaded by ml_engine.py's
                      predict_ct_risk() the next time it runs.

Requires: pip install torch torchvision pandas scikit-learn pillow openpyxl
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as tv_transforms
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.exit(
        "ERROR: this script requires torch + torchvision. "
        "Install with: pip install torch torchvision"
    )

from sklearn.model_selection import train_test_split

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: this script requires Pillow. Install with: pip install pillow")

_IMAGE_SIZE = 224


class CTDataset(Dataset):
    def __init__(self, image_dir, filenames, labels, transform):
        self.image_dir = image_dir
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.filenames[idx])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label


def load_labels(
    labels_path: str,
    filename_col: str,
    label_col: str,
    build_filename_from_id: bool = False,
    pad_width: int = 3,
    extension: str = ".png",
):
    ext = os.path.splitext(labels_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(labels_path)
    else:
        df = pd.read_csv(labels_path)

    for col in (filename_col, label_col):
        if col not in df.columns:
            sys.exit(f"ERROR: column '{col}' not found. Available columns: {list(df.columns)}")

    df = df[[filename_col, label_col]].dropna()

    # Coerce label column to 0/1, whatever its native type.
    raw = df[label_col]
    unique_vals = raw.unique()
    if set(pd.unique(raw.astype(str).str.lower())) <= {"0", "1", "true", "false"}:
        mapping = {"0": 0, "false": 0, "1": 1, "true": 1}
        labels = raw.astype(str).str.lower().map(mapping).astype(int).to_numpy()
    elif len(unique_vals) == 2:
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        print(f"Label mapping inferred: {mapping}")
        labels = raw.map(mapping).astype(int).to_numpy()
    else:
        sys.exit(
            f"ERROR: label column '{label_col}' has {len(unique_vals)} unique values "
            f"({unique_vals[:10]}...). This script expects a binary label. "
            "Binarize it yourself first (e.g. thresholding) and retry."
        )

    if build_filename_from_id:
        # e.g. id=7, pad_width=3, extension=".png" -> "007.png"
        filenames = (
            df[filename_col]
            .astype(int)
            .apply(lambda i: f"{i:0{pad_width}d}{extension}")
            .to_numpy()
        )
    else:
        filenames = df[filename_col].astype(str).to_numpy()

    return filenames, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Folder containing the CT images")
    parser.add_argument("--labels", required=True, help="CSV or Excel file with filename+label columns")
    parser.add_argument("--filename-col", required=True, help="Column name holding image filenames (or numeric ids -- see --build-filename-from-id)")
    parser.add_argument("--label-col", required=True, help="Column name holding the binary label")
    parser.add_argument(
        "--build-filename-from-id",
        action="store_true",
        help="Treat --filename-col as a bare numeric id and build the actual "
        "filename as zero-padded-id + extension (e.g. id=7 -> '007.png'). "
        "Use this when the CSV has no file extension in its filename column.",
    )
    parser.add_argument("--pad-width", type=int, default=3, help="Zero-padding width when --build-filename-from-id is set (default: 3)")
    parser.add_argument("--extension", default=".png", help="Image extension when --build-filename-from-id is set (default: .png)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write ct_resnet.pt into (default: this script's folder)",
    )
    args = parser.parse_args()

    filenames, labels = load_labels(
        args.labels,
        args.filename_col,
        args.label_col,
        build_filename_from_id=args.build_filename_from_id,
        pad_width=args.pad_width,
        extension=args.extension,
    )
    print(f"Loaded {len(labels)} labeled images | positive rate: {labels.mean():.3%}")

    # Drop rows whose image file doesn't actually exist (common with
    # partially-downloaded or mismatched datasets).
    keep = np.array([os.path.isfile(os.path.join(args.image_dir, f)) for f in filenames])
    if not keep.all():
        print(f"WARNING: {(~keep).sum()} listed files not found in {args.image_dir}, skipping them.")
    filenames, labels = filenames[keep], labels[keep]

    X_train, X_val, y_train, y_val = train_test_split(
        filenames, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = tv_transforms.Compose(
        [
            tv_transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = CTDataset(args.image_dir, X_train, y_train, train_transform)
    val_ds = CTDataset(args.image_dir, X_val, y_val, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    try:
        model = tv_models.resnet18(weights="DEFAULT")
    except Exception:
        model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # more numerically stable than sigmoid+BCELoss

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    out_path = os.path.join(args.out_dir, "ct_resnet.pt")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved trained ResNet18 weights -> {out_path}")
    print("ml_engine.py will now load these automatically on next run.")


if __name__ == "__main__":
    main()
