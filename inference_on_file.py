# inference_on_file.py
# batch inference on a folder or CSV of image paths
# If the user provides a ground-truth CSV, evaluation graphs are produced.
# this file will be called from client.py
# or can be run directly: python inference_on_file.py

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

import config
from inference_one import _load_model, _get_transform, predict_image


def collect_images(source: Path) -> list[Path]:
    """
    Return all image paths from:
      - a folder (recursive scan), or
      - a CSV that has a 'filepath' column  (train/val/test split CSVs), or
      - a CSV/text file with image paths in the first column.
    """
    if source.is_dir():
        imgs = sorted(
            p for p in source.rglob("*")
            if p.suffix.lower() in config.IMG_EXTS
        )
        print(f"  Found {len(imgs):,} images in {source}")
        return imgs

    suffix = source.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(source)

        # Preferred way is to use the 'filepath' column
        if "filepath" in df.columns:
            paths  = []
            broken = []
            for raw in df["filepath"]:
                p = Path(str(raw))
                if p.exists():
                    paths.append(p)
                else:
                    broken.append(p)

            if broken:
                print(f"  {len(broken):,} filepaths in CSV not found on disk")
                print(f"     First missing: {broken[0]}")
                print(f"     Check that the dataset folder has not moved since")
                print(f"     preprocessing was run.  Expected folder:")
                print(f"       {config.IMG_ROOT_256}")

            print(f"  Found {len(paths):,} images via 'filepath' column in {source.name}")
            return paths

        # For fallback we will treat first column as paths 
        first_col = df.columns[0]
        paths, broken = [], []
        for raw in df[first_col]:
            raw = str(raw).strip()
            if not raw or raw.lower() == first_col.lower():
                continue
            p = Path(raw)
            if p.exists():
                paths.append(p)
            else:
                broken.append(raw)

        if broken:
            print(f" {len(broken):,} entries in column '{first_col}' "
                  f"could not be resolved as paths.")
            print(f"     First broken: {broken[0]}")

        print(f"  Found {len(paths):,} images via '{first_col}' column in {source.name}")
        return paths

    # Plain text file where one path per line
    paths, broken = [], []
    for line in source.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if p.exists():
            paths.append(p)
        else:
            broken.append(line)

    if broken:
        print(f" {len(broken):,} lines could not be resolved as paths.")

    print(f" Found {len(paths):,} images from {source.name}")
    return paths


def load_ground_truth(gt_path: Path) -> dict[str, int]:
    """
    Load a ground-truth CSV. Handles three formats:
      1. Our split CSVs  — has 'image' stem + integer 'label' column
      2. Original ISIC   — has 'image' stem + one-hot CLASS_NAME columns
      3. Simple          — has 'image' stem + 'label' column with class name strings
    Returns {image_stem: integer_label}.
    """
    df  = pd.read_csv(gt_path)
    id_col = df.columns[0]     # always the image identifier

    # Format 1 — integer label column
    if "label" in df.columns and pd.api.types.is_integer_dtype(df["label"]):
        result = {}
        for _, row in df.iterrows():
            stem = Path(str(row[id_col])).stem   # handle full path or just stem
            lbl  = int(row["label"])
            if 0 <= lbl < config.NUM_CLASSES:
                result[stem] = lbl
        print(f"  Detected format: split CSV  (integer labels)")
        return result

    # Format 2 — one-hot columns matching CLASS_NAMES
    if all(c in df.columns for c in config.CLASS_NAMES):
        result = {}
        for _, row in df.iterrows():
            stem = Path(str(row[id_col])).stem
            lbl  = int(pd.Series(
                [row[c] for c in config.CLASS_NAMES]
            ).values.argmax())
            result[stem] = lbl
        print(f"  Detected format: one-hot columns")
        return result

    # Format 3 — string label column
    if "label" in df.columns:
        name_to_idx = {n: i for i, n in enumerate(config.CLASS_NAMES)}
        result = {}
        for _, row in df.iterrows():
            stem = Path(str(row[id_col])).stem
            lbl  = name_to_idx.get(str(row["label"]))
            if lbl is not None:
                result[stem] = lbl
        print(f"  Detected format: string label column")
        return result

    raise ValueError(
        "Could not detect ground-truth format\n"
        "Expected one of:\n"
        "  - an integer 'label' column  (train/val/test_split.csv)\n"
        "  - one-hot columns matching CLASS_NAMES\n"
        "  - a string 'label' column with class name values"
    )


# ── Evaluation graphs ─────────────────────────────────────────────────────────

def save_eval_graphs(all_labels, all_preds, all_probs, out_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc,
        precision_recall_curve, average_precision_score,
    )
    from sklearn.preprocessing import label_binarize
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    num_cls    = config.NUM_CLASSES
    labels_bin = label_binarize(all_labels, classes=list(range(num_cls)))

    print(f"\n  Saving evaluation graphs to {out_dir} …")

    # 1) Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=config.CLASS_NAMES, digits=4)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    print("\n  Classification Report:")
    print(report)

    # 2) Confusion matrices
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix — Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cmap="Blues", vmin=0, vmax=1, ax=axes[1])
    axes[1].set_title("Confusion Matrix — Row-Normalised (Recall)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close()
    print(" confusion_matrix.png")

    # 3) Per-class accuracy bar
    per_cls_acc = []
    for i in range(num_cls):
        mask = all_labels == i
        per_cls_acc.append((all_preds[mask] == i).mean() if mask.sum() else 0.0)

    colors = ["#d62728" if a < 0.6 else "#ff7f0e" if a < 0.8 else "#2ca02c"
              for a in per_cls_acc]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(config.CLASS_NAMES, per_cls_acc, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(np.mean(per_cls_acc), color="steelblue", linestyle="--",
               linewidth=1.5, label=f"Mean: {np.mean(per_cls_acc):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    for bar, acc in zip(bars, per_cls_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    legend_els = [
        Patch(facecolor="#d62728", label="< 0.60"),
        Patch(facecolor="#ff7f0e", label="0.60 – 0.80"),
        Patch(facecolor="#2ca02c", label="≥ 0.80"),
    ]
    ax.legend(handles=legend_els, loc="lower right", title="Accuracy band")
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_accuracy.png", dpi=120)
    plt.close()
    print(" per_class_accuracy.png")

    # 4) ROC curves
    fig, ax = plt.subplots(figsize=(9, 7))
    auc_scores = {}
    colors_roc = plt.cm.tab10(np.linspace(0, 1, num_cls))
    for i, (cls, color) in enumerate(zip(config.CLASS_NAMES, colors_roc)):
        if labels_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc     = auc(fpr, tpr)
        auc_scores[cls] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{cls}  (AUC = {roc_auc:.3f})")
    macro_auc = np.mean(list(auc_scores.values())) if auc_scores else 0.0
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_title(f"ROC Curves — One-vs-Rest  (Macro AUC = {macro_auc:.3f})")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=120)
    plt.close()
    print(" roc_curves.png")

    # 5) Precision-Recall curves
    fig, ax = plt.subplots(figsize=(9, 7))
    ap_scores = {}
    colors_pr = plt.cm.tab10(np.linspace(0, 1, num_cls))
    for i, (cls, color) in enumerate(zip(config.CLASS_NAMES, colors_pr)):
        if labels_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(labels_bin[:, i], all_probs[:, i])
        ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        ap_scores[cls] = ap
        ax.plot(rec, prec, color=color, lw=1.8, label=f"{cls}  (AP = {ap:.3f})")
    mean_ap = np.mean(list(ap_scores.values())) if ap_scores else 0.0
    ax.set_title(f"Precision-Recall Curves  (Mean AP = {mean_ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=120)
    plt.close()
    print(" pr_curves.png")

    print(f"\n  Macro AUC : {macro_auc:.4f}")
    print(f"  Mean AP   : {mean_ap:.4f}")
    print(f"  Overall acc: {(all_preds == all_labels).mean():.4f}")


# Main
def run():
    if config.BASE_PATH is None:
        raise RuntimeError("Paths not set. Run client.py first.")

    if not config.CKPT_PATH.exists():
        print(f"\n No checkpoint at {config.CKPT_PATH}. Train first.")
        return

    print("\n" + "=" * 60)
    print("  BATCH INFERENCE")
    print("=" * 60)

    # Source
    src_raw = input(
        "\n  Enter path to image folder or a CSV/text file of image paths:\n  > "
    ).strip()
    source = Path(src_raw)
    if not source.exists():
        print(f" Path not found: {source}")
        return

    img_paths = collect_images(source)
    if not img_paths:
        print(" No images found.")
        return

    # Ground truth?
    use_gt = input("\n  Do you have a ground-truth CSV? [y/N]: ").strip().lower() == "y"
    gt_map = {}
    if use_gt:
        gt_raw = input("  Enter path to ground-truth CSV: ").strip()
        gt_path = Path(gt_raw)
        if not gt_path.exists():
            print(f" File not found: {gt_path}. Proceeding without ground truth")
            use_gt = False
        else:
            try:
                gt_map = load_ground_truth(gt_path)
                print(f" Loaded {len(gt_map):,} ground-truth labels")
            except Exception as e:
                print(f" Could not load ground truth: {e}")
                use_gt = False

    # Output directory for results
    default_out = str(source.parent / "dermaboss_results")
    out_raw     = input(f"\n  Output directory [{default_out}]: ").strip()
    out_dir     = Path(out_raw) if out_raw else Path(default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(device)
    transform   = _get_transform()
    print(f"\n  Checkpoint : epoch {ckpt['epoch']}  val_acc {ckpt['val_acc']:.4f}")
    print(f"  Device     : {device}")
    print(f"  Images     : {len(img_paths):,}")
    print()

    # Inference
    rows       = []
    all_labels = []
    all_preds  = []
    all_probs  = []

    for img_path in tqdm(img_paths, desc="  Inferring", unit="img", ncols=70):
        try:
            result = predict_image(img_path, model, transform, device)
        except Exception as e:
            print(f"\n Skipping {img_path.name}: {e}")
            continue

        row = {
            "image"           : img_path.name,
            "predicted_class" : result["predicted_class"],
            "confidence"      : round(result["confidence"], 4),
        }
        for cls, prob in result["all_probs"].items():
            row[f"prob_{cls}"] = round(prob, 4)

        stem = img_path.stem
        if use_gt and stem in gt_map:
            row["true_class"] = config.CLASS_NAMES[gt_map[stem]]
            row["correct"]    = int(result["predicted_class"] == row["true_class"])
            all_labels.append(gt_map[stem])
            all_preds.append(config.CLASS_NAMES.index(result["predicted_class"]))
            all_probs.append([result["all_probs"][c] for c in config.CLASS_NAMES])

        rows.append(row)

    # Save predictions CSV
    pred_csv = out_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    print(f"\n Predictions saved → {pred_csv}")

    # Evaluation graphs (only if ground truth provided and matched)
    if use_gt and all_labels:
        matched = len(all_labels)
        print(f"\n  Ground-truth matched for {matched:,} / {len(rows):,} images.")
        save_eval_graphs(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            out_dir,
        )
    elif use_gt:
        print("\n No ground-truth labels matched image stems. "
              "Check that the CSV image column matches filenames.")


if __name__ == "__main__":
    if config.BASE_PATH is None:
        path = input("Enter dataset base directory: ").strip()
        config.set_paths(Path(path))
    run()