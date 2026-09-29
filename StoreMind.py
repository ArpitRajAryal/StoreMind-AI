from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

SHELF_LABELS = ("empty", "low", "stocked")


def build_shelf_tasks(inventory, observations, confidence_min=.85):
    required_i = {"sku", "product", "on_hand", "verified_backroom_units", "shelf_min"}
    required_o = {"sku", "shelf_id", "observed_shelf_units", "confidence", "observed_at"}
    if not required_i.issubset(inventory.columns) or not required_o.issubset(observations.columns):
        raise ValueError("Missing required inventory or observation columns")
    if inventory.sku.duplicated().any() or observations.sku.duplicated().any():
        raise ValueError("Expected one inventory and one latest observation row per SKU")
    merged = observations.merge(inventory, on="sku", how="left", validate="one_to_one", indicator=True)
    if (merged._merge != "both").any():
        raise ValueError("Observation contains an unknown SKU")
    tasks = []
    for row in merged.itertuples(index=False):
        if not 0 <= row.confidence <= 1 or row.observed_shelf_units < 0 or row.on_hand < 0:
            raise ValueError(f"Invalid observation or stock count for {row.sku}")
        backroom = row.verified_backroom_units
        if row.confidence < confidence_min:
            action, reason = "VERIFY_OBSERVATION", "Observation confidence is too low"
        elif row.observed_shelf_units == 0 and row.on_hand == 0:
            action, reason = "PROPOSE_REORDER", "Shelf is empty and system stock is zero"
        elif row.observed_shelf_units < row.shelf_min and pd.notna(backroom) and backroom > 0:
            action, reason = "RESTOCK_FROM_BACKROOM", "Visible shelf is below minimum; backroom stock was verified"
        elif row.observed_shelf_units == 0 and row.on_hand > 0:
            action, reason = "CHECK_SHELF_AND_BACKROOM", "System shows stock but shelf appears empty"
        elif row.observed_shelf_units < row.shelf_min:
            action, reason = "VERIFY_LOW_SHELF", "Visible shelf is below minimum"
        else:
            action, reason = "OK", "No shelf task indicated"
        tasks.append({"sku":row.sku, "product":row.product, "shelf_id":row.shelf_id,
                      "action":action, "reason":reason, "observed_at":row.observed_at,
                      "confidence":row.confidence})
    return pd.DataFrame(tasks)


def review_movements(movements, pos, window_minutes=30):
    required_m = {"movement_id", "sku", "units", "at", "camera_ref"}
    required_p = {"sale_id", "sku", "units", "at"}
    if not required_m.issubset(movements.columns) or not required_p.issubset(pos.columns):
        raise ValueError("Missing required movement or POS columns")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    window = pd.Timedelta(minutes=window_minutes)
    used_sales = set()
    reviews = []
    for m in movements.sort_values("at").itertuples(index=False):
        if m.units <= 0:
            raise ValueError("Movement units must be positive")
        candidates = pos.loc[(pos["sku"] == m.sku) & (pos["units"] >= m.units) &
                             (pos["at"] >= m.at) & (pos["at"] <= m.at + window) &
                             (~pos["sale_id"].isin(used_sales))].sort_values("at")
        sale_id = candidates.iloc[0].sale_id if len(candidates) else None
        if sale_id is not None:
            used_sales.add(sale_id)
        reviews.append({"movement_id":m.movement_id, "sku":m.sku,
                        "status":"MATCHED_POS" if sale_id else "REVIEW_UNMATCHED_MOVEMENT",
                        "sale_id":sale_id, "camera_ref":m.camera_ref, "at":m.at,
                        "next_step":"No incident task" if sale_id else "Staff verify POS, transfers, returns, and footage"})
    return pd.DataFrame(reviews)


def validate_manifest(csv_path):
    from PIL import Image
    df = pd.read_csv(csv_path)
    expected = {"image_path", "sku", "shelf_id", "capture_session", "label"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Manifest needs columns: {sorted(expected)}")
    if len(df) == 0 or df[list(expected)].isna().any().any():
        raise ValueError("Manifest is empty or contains missing required values")
    if not set(df.label).issubset(SHELF_LABELS):
        raise ValueError(f"Labels must be in {SHELF_LABELS}")
    base = Path(csv_path).resolve().parent
    for raw in df.image_path:
        path = Path(raw)
        path = path if path.is_absolute() else base / path
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path) as img: img.verify()
    if df.capture_session.nunique() < 2:
        raise ValueError("At least two capture sessions are needed for a grouped split")
    return df


def train_shelf_state_model(manifest_csv, epochs=3, batch_size=16, seed=42):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import classification_report, confusion_matrix
    from PIL import Image

    frame = validate_manifest(manifest_csv).reset_index(drop=True)
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    train_idx, val_idx = next(split.split(frame, frame.label, groups=frame.capture_session))
    train_frame, val_frame = frame.iloc[train_idx], frame.iloc[val_idx]
    if set(train_frame.label) != set(SHELF_LABELS):
        raise ValueError("Training split must contain all three classes; collect more sessions")
    root = Path(manifest_csv).resolve().parent
    weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
    ])
    class ShelfDataset(Dataset):
        def __init__(self, rows, transform): self.rows, self.transform = rows.reset_index(drop=True), transform
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows.iloc[idx]
            p = Path(row.image_path)
            if not p.is_absolute(): p = root / p
            with Image.open(p) as image: x = self.transform(image.convert("RGB"))
            return x, SHELF_LABELS.index(row.label)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=weights)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(SHELF_LABELS))
    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(ShelfDataset(train_frame, augment), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ShelfDataset(val_frame, preprocess), batch_size=batch_size)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"Epoch {epoch + 1}/{epochs}: training loss {total_loss / len(train_frame):.3f}")
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x.to(device)).argmax(1).cpu().tolist()
            predicted.extend(pred)
            truth.extend(y.tolist())
    print(classification_report(truth, predicted, labels=range(3), target_names=SHELF_LABELS, zero_division=0))
    print("Confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(truth, predicted, labels=range(3)))
    return model


def preserve_review_clip(video_path, event_offset_seconds, output_path, before=10, after=15):
    import shutil
    import subprocess
    source, target = Path(video_path), Path(output_path)
    if not source.is_file(): raise FileNotFoundError(source)
    if shutil.which("ffmpeg") is None: raise RuntimeError("ffmpeg is required")
    if event_offset_seconds < 0 or before < 0 or after <= 0:
        raise ValueError("Invalid clip time range")
    start = max(0, float(event_offset_seconds) - before)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start),
                    "-i", str(source), "-t", str(before + after), "-c:v", "libx264",
                    "-c:a", "aac", str(target)], check=True)
    return target


def record_review(movement_id, reviewer, resolution, note=""):
    allowed = {"sale_found", "staff_transfer", "return_or_restock", "unclear", "other"}
    if not reviewer.strip() or resolution not in allowed:
        raise ValueError(f"Reviewer and one of {sorted(allowed)} are required")
    return {"movement_id":movement_id, "reviewer":reviewer, "resolution":resolution,
            "note":note, "reviewed_at":pd.Timestamp.now(tz="UTC")}


def main():
    parser = argparse.ArgumentParser(description="StoreMind retail operations prototype")
    parser.add_argument("--manifest", type=Path, help="CSV of real labeled shelf images; requires torch and torchvision")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--chart", type=Path, help="Save the simulated shelf task chart to this image path")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    print("StoreMind prototype: SIMULATED operational records")
    as_of = pd.Timestamp("2026-09-29 14:12:00+10:00")
    inventory = pd.DataFrame([
        {"sku":"HP-101", "product":"Headphones", "on_hand":4, "verified_backroom_units":pd.NA, "shelf_min":2},
        {"sku":"CB-202", "product":"USB-C cable", "on_hand":0, "verified_backroom_units":0, "shelf_min":3},
        {"sku":"WB-303", "product":"Water bottle", "on_hand":7, "verified_backroom_units":5, "shelf_min":4},
        {"sku":"PH-404", "product":"Phone case", "on_hand":5, "verified_backroom_units":2, "shelf_min":2},
    ])
    observations = pd.DataFrame([
        {"sku":"HP-101", "shelf_id":"A1", "observed_shelf_units":0, "confidence":.94, "observed_at":as_of},
        {"sku":"CB-202", "shelf_id":"B2", "observed_shelf_units":0, "confidence":.97, "observed_at":as_of},
        {"sku":"WB-303", "shelf_id":"C3", "observed_shelf_units":2, "confidence":.93, "observed_at":as_of},
        {"sku":"PH-404", "shelf_id":"D4", "observed_shelf_units":3, "confidence":.91, "observed_at":as_of},
    ])
    movements = pd.DataFrame([
        {"movement_id":"M-001", "sku":"HP-101", "units":1, "at":as_of-pd.Timedelta(minutes=8), "camera_ref":"cam-A1-1410"},
        {"movement_id":"M-002", "sku":"PH-404", "units":1, "at":as_of-pd.Timedelta(minutes=6), "camera_ref":"cam-D4-1406"},
    ])
    pos = pd.DataFrame([
        {"sale_id":"S-001", "sku":"PH-404", "units":1, "at":as_of-pd.Timedelta(minutes=4)},
    ])
    shelf_tasks = build_shelf_tasks(inventory, observations)
    movement_reviews = review_movements(movements, pos)
    staff_queue = shelf_tasks.loc[shelf_tasks.action != "OK", ["sku", "product", "shelf_id", "action", "reason"]].copy()
    incident_queue = movement_reviews.loc[movement_reviews.status == "REVIEW_UNMATCHED_MOVEMENT", ["sku", "camera_ref", "status", "next_step"]].copy()
    print("\nSIMULATED inventory")
    print(inventory.to_string(index=False))
    print("\nSIMULATED shelf observations")
    print(observations.to_string(index=False))
    print("\nStaff tasks")
    print(staff_queue.to_string(index=False))
    print("\nMovement review queue")
    print(incident_queue.to_string(index=False))

    assert shelf_tasks.set_index("sku").loc["HP-101", "action"] == "CHECK_SHELF_AND_BACKROOM"
    assert shelf_tasks.set_index("sku").loc["CB-202", "action"] == "PROPOSE_REORDER"
    assert shelf_tasks.set_index("sku").loc["WB-303", "action"] == "RESTOCK_FROM_BACKROOM"
    assert movement_reviews.set_index("movement_id").loc["M-002", "status"] == "MATCHED_POS"
    assert movement_reviews.set_index("movement_id").loc["M-001", "status"] == "REVIEW_UNMATCHED_MOVEMENT"
    print("\nPrototype checks passed.")

    if args.chart:
        import matplotlib.pyplot as plt
        counts = shelf_tasks.action.value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.barh(counts.index.str.replace("_", " "), counts.values,
                color=["#3D8F9D" if a == "OK" else "#F0A44B" for a in counts.index])
        ax.set(xlabel="Number of simulated SKUs", title="Shelf task distribution — sample data")
        ax.set_xlim(0, max(counts.values) + .5)
        for i, count in enumerate(counts.values):
            ax.text(count + .03, i, str(count), va="center")
        fig.tight_layout()
        args.chart.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.chart, dpi=160)
        plt.close(fig)
        print(f"Saved chart: {args.chart}")

    if args.manifest:
        real_manifest = validate_manifest(args.manifest)
        print(f"Validated {len(real_manifest)} real labeled shelf photos")
        train_shelf_state_model(args.manifest, epochs=args.epochs)


if __name__ == "__main__":
    main()
