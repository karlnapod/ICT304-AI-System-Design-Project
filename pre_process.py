# pre_process.py — resize all images to 256 px, then save train/val/test CSVs
# Can be imported and called via pre_process.run()
# or run directly: python pre_process.py

import json
import datetime
import os
from pathlib import Path

import pandas as pd
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def split_and_save_csvs():
    """
    Loading the ground-truth CSV, dropping UNK label, index resized images,
    perform the 80/10/10 stratified split, and save three CSVs:
      train_split.csv
      val_split.csv
      test_split.csv
    """
    print("\n  Building train / val / test CSVs …")

    # Loading the CSV and dropping UNK class
    df = pd.read_csv(config.GT_CSV)
    all_cols = config.CLASS_NAMES + [config.DROP_CLASS]

    assert (df[all_cols].sum(axis=1) == 1).all(), \
        "Ground-truth CSV is not strictly one-hot"

    df = df[df[config.DROP_CLASS] != 1].reset_index(drop=True)
    df["label"] = df[config.CLASS_NAMES].values.argmax(axis=1)

    # Indexing the resized images
    stem_to_path = {
        p.stem: str(p)
        for p in config.IMG_ROOT_256.rglob("*")
        if p.suffix.lower() in config.IMG_EXTS
    }
    df["filepath"] = df["image"].map(stem_to_path)
    missing = df["filepath"].isna().sum()
    if missing:
        print(f" {missing:,} rows had no matching image hence dropped.")
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)

    total = len(df)
    print(f"  Total usable samples: {total:,}")

    # 80 / 10 / 10 stratified split
    train_val, test_df = train_test_split(
        df,
        test_size=config.TEST_FRACTION,
        random_state=config.RANDOM_SEED,
        stratify=df["label"],
    )
    val_relative = config.VAL_FRACTION / (1.0 - config.TEST_FRACTION)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=config.RANDOM_SEED,
        stratify=train_val["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # Saving the CSVs
    out_cols = ["image", "filepath", "label"] + config.CLASS_NAMES

    train_path = config.BASE_PATH / "train_split.csv"
    val_path   = config.BASE_PATH / "val_split.csv"
    test_path  = config.BASE_PATH / "test_split.csv"

    train_df[out_cols].to_csv(train_path, index=False)
    val_df[out_cols].to_csv(val_path,     index=False)
    test_df[out_cols].to_csv(test_path,   index=False)

    # Printing the summary stats
    print()
    print(f"  {'Split':<8}  {'Samples':>8}  {'%':>6}  {'CSV'}")
    print(f"  {'─'*55}")
    for name, split_df, path in [
        ("Train", train_df, train_path),
        ("Val",   val_df,   val_path),
        ("Test",  test_df,  test_path),
    ]:
        pct = len(split_df) / total * 100
        print(f"  {name:<8}  {len(split_df):>8,}  {pct:>5.1f}%  {path.name}")

    print()
    print("  Class balance across splits:")
    print(f"  {'Class':<8}", end="")
    for name in ["Train", "Val", "Test"]:
        print(f"  {name:>8}", end="")
    print()
    for i, cls in enumerate(config.CLASS_NAMES):
        print(f"  {cls:<8}", end="")
        for split_df in [train_df, val_df, test_df]:
            pct = (split_df["label"] == i).sum() / len(split_df) * 100
            print(f"  {pct:>7.2f}%", end="")
        print()

    return train_path, val_path, test_path


def run():
    if config.DONE_FLAG is None:
        raise RuntimeError(
            "Paths not set. Run client.py or call config.set_paths() first"
        )

    # checking if preprocessing is already done
    if config.DONE_FLAG.exists():
        meta = json.loads(config.DONE_FLAG.read_text())
        print(f"\nPre-resize already completed on {meta['date']}")
        print(f"    {meta['total']:,} images  |  errors: {meta['errors']}")

        # Checking if CSVs were saved in a previous run
        csvs_exist = all(
            (config.BASE_PATH / f).exists()
            for f in ("train_split.csv", "val_split.csv", "test_split.csv")
        )
        if csvs_exist:
            print("Split CSVs already exist and are assumed to be correct.")
            return
        else:
            print("Split CSVs not found — regenerating …")
            split_and_save_csvs()
            return

    # Scanning the source images 
    all_imgs = [
        p for p in config.IMG_ROOT.rglob("*")
        if p.suffix.lower() in config.IMG_EXTS
    ]
    total = len(all_imgs)

    if total == 0:
        print(f"\nNo images found under {config.IMG_ROOT}")
        print("\nCheck the path and re-run.")
        return

    print(f"\n  {'─'*56}")
    print(f"  Pre-processing: resize to {config.RESIZE_PX} px max side")
    print(f"  Source  : {config.IMG_ROOT}")
    print(f"  Dest    : {config.IMG_ROOT_256}")
    print(f"  Images  : {total:,}")
    print(f"  Safe to interrupt — already-done files are skipped on re-run.")
    print(f"  {'─'*56}\n")

    skipped, resized, errors = 0, 0, []

    for src in tqdm(all_imgs, unit="img", desc="  Resizing", ncols=70):
        rel  = src.relative_to(config.IMG_ROOT)
        dest = config.IMG_ROOT_256 / rel.with_suffix(".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            skipped += 1
            continue

        try:
            img = Image.open(src).convert("RGB")
            img.thumbnail((config.RESIZE_PX, config.RESIZE_PX), Image.LANCZOS)
            img.save(dest, "JPEG", quality=95)
            resized += 1
        except Exception as e:
            errors.append((str(src), str(e)))

    # Writing completion flag
    config.DONE_FLAG.write_text(json.dumps({
        "date"   : datetime.datetime.now().isoformat(timespec="seconds"),
        "total"  : resized + skipped,
        "resized": resized,
        "skipped": skipped,
        "errors" : len(errors),
    }, indent=2))

    print(f"\nResize complete.")
    print(f"    Resized : {resized:,}")
    print(f"    Skipped : {skipped:,}  (already existed)")
    print(f"    Errors  : {len(errors)}")
    if errors:
        print("    First 5 errors:")
        for p, e in errors[:5]:
            print(f"      {p}  →  {e}")

    # finally build and save the split CSVs
    split_and_save_csvs()
    print("\nPre-processing complete.")


if __name__ == "__main__":
    if config.BASE_PATH is None:
        path = input("Enter dataset base directory: ").strip()
        config.set_paths(Path(path))
    run()