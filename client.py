# client.py will be used as the orchestrator
# the file can be run with: python client.py

import sys
import os
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import config


def separator(title: str = ""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def ask(prompt: str, default: str = "") -> str:
    # Print prompt and return stripped input, or default on empty Enter.
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def confirm(prompt: str) -> bool:
    return ask(prompt + " [y/N]").lower() == "y"


def download_with_progress(url: str, dest: Path, label: str):
    # urllib download with a simple terminal progress bar.
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} …")

    def hook(block, bsize, total):
        done  = block * bsize
        pct   = min(done / total * 100, 100) if total > 0 else 0
        bar   = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        mb    = done / 1024**2
        total_mb = total / 1024**2
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print(f"\r Saved → {dest}  ({dest.stat().st_size / 1024**2:.1f} MB)")


# Locate/Download the dataset
def resolve_dataset() -> Path:
    """
    Check whether the dataset exists.
    If not, offer to:
      - provide a custom path,
    or
      - download automatically (~9 GB warning included).
    Returns the confirmed base Path.
    """
    separator("DATASET SETUP")

    URLS = {
        "gt_csv"  : "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
        "meta_csv": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv",
        "images"  : "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip",
    }

    # Ask for base directory
    default_base = str(Path.home() / "dermaboss_data")
    base_str     = ask("Enter dataset base directory", default_base)
    base         = Path(base_str)
    config.set_paths(base)

    gt_csv    = config.GT_CSV
    img_root  = config.IMG_ROOT
    zip_path  = base / "ISIC_2019_Training_Input.zip"

    # checking
    csv_ok = gt_csv.exists()
    imgs_ok = img_root.exists() and any(img_root.rglob("*.jpg"))

    print()
    print(f"GT CSV: {'found' if csv_ok  else 'missing'}  ({gt_csv})")
    print(f"Images: {'found' if imgs_ok else 'missing'}  ({img_root})")

    if csv_ok and imgs_ok:
        img_count = sum(1 for _ in img_root.rglob("*.jpg"))
        print(f"\n Dataset looks complete ({img_count:,} images found)")
        return base

    # incase something is missing
    print("\n  Some dataset files are missing")
    print("Options:")
    print("1) I already have the files — let me provide the correct path")
    print("2) Download automatically   (GT CSV ~1 MB, Images ZIP ~9 GB)")
    print("3) Exit and handle manually")

    choice = ask("Choose", "1").strip()

    if choice == "1":
        new_path = ask("Enter the correct base directory path")
        base     = Path(new_path)
        config.set_paths(base)
        # Re-check
        if not config.GT_CSV.exists():
            print(f"\n Still cannot find GT CSV at {config.GT_CSV}")
            print(" Please verify the path and re-run client.py")
            sys.exit(1)
        print("Path accepted")
        return base

    elif choice == "2":
        print()
        # Download CSV if missing
        if not csv_ok:
            download_with_progress(URLS["gt_csv"],   config.GT_CSV,  "GroundTruth CSV")
            download_with_progress(URLS["meta_csv"], config.META_CSV, "Metadata CSV")

        # Download + extract images if missing
        if not imgs_ok:
            print("\n The image ZIP is approximately 9 GB.")
            print("!!This may take 15–45 minutes depending on your connection!!")
            if not confirm("  Proceed with download?"):
                print("  Download cancelled, Exiting")
                sys.exit(0)

            if not zip_path.exists():
                download_with_progress(URLS["images"], zip_path, "ISIC 2019 Images ZIP")
            else:
                print(f"  ZIP already present: {zip_path}")

            print(f"\nExtracting ZIP -> {base} …")
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                for idx, member in enumerate(members):
                    zf.extract(member, base)
                    pct = (idx + 1) / len(members) * 100
                    print(f"\r  Extracting … {pct:.1f}%  ({idx+1:,}/{len(members):,} files)",
                          end="", flush=True)
            print("\n Extraction complete")

            if confirm("  Delete ZIP to save disk space?"):
                zip_path.unlink()
                print("  ZIP deleted")

        return base

    else:
        print("  Exiting. Re-run client.py when ready")
        sys.exit(0)


# preprocessing check

def check_preprocessing():
    separator("PREPROCESSING CHECK")

    resize_done = config.DONE_FLAG.exists()
    csvs_done   = all(
        (config.BASE_PATH / f).exists()
        for f in ("train_split.csv", "val_split.csv", "test_split.csv")
    )

    if resize_done:
        meta = json.loads(config.DONE_FLAG.read_text())
        print(f"Images resized on {meta['date']}")
        print(f"{meta['total']:,} images at 256 px  (errors: {meta['errors']})")
    else:
        img_count = sum(1 for _ in config.IMG_ROOT.rglob("*.jpg"))
        print(f"Images not yet resized  ({img_count:,} raw images found)")

    if csvs_done:
        for name in ("train_split.csv", "val_split.csv", "test_split.csv"):
            rows = sum(1 for _ in open(config.BASE_PATH / name)) - 1  # minus header
            print(f"{name}  ({rows:,} rows)")
    else:
        print(f"Split CSVs not found  "
              f"(train_split.csv / val_split.csv / test_split.csv)")

    if resize_done and csvs_done:
        return

    # if something is missing then offer to run preprocessing
    print()
    if not resize_done:
        print("Pre-processing is required before training")
        print("It resizes all images to 256 px and saves the train/val/test splits")
        print("This is a one-time step (~10–20 min)")
    else:
        print("The split CSVs are missing but images are already resized")
        print("Re-running preprocessing will rebuild the CSVs immediately (no resize)")

    if confirm("  Run preprocessing now?"):
        import pre_process
        pre_process.run()
    else:
        print("Skipping — you can run it later from the main menu (option 1).")
        if not resize_done:
            print(" Training will fail until preprocessing is complete.")


# main menu
def main_menu():
    separator("DERMABOSS — MAIN MENU")
    print("  1) Preprocess dataset (resize to 256 px)")
    print("  2) Train model")
    print("  3) Inference — single image")
    print("  4) Inference — batch file / folder  (optional ground-truth CSV)")
    print("  5) Exit")
    print()

    while True:
        choice = ask("Select option", "5").strip()

        if choice == "1":
            import pre_process
            pre_process.run()

        elif choice == "2":
            import train
            train.run()

        elif choice == "3":
            import inference_one
            inference_one.run()

        elif choice == "4":
            import inference_on_file
            inference_on_file.run()

        elif choice == "5":
            print("\n  Goodbye.")
            sys.exit(0)

        else:
            print("  Invalid choice — enter 1–5.")

        # After returning from an option, show menu again
        separator("DERMABOSS — MAIN MENU")
        print("  1) Preprocess dataset")
        print("  2) Train model")
        print("  3) Inference — single image")
        print("  4) Inference — batch file / folder")
        print("  5) Exit")
        print()



if __name__ == "__main__":
    print("=" * 60)
    print("  DermaBoss — Skin Lesion Classifier")
    print("=" * 60)

    base = resolve_dataset()
    config.set_paths(base)
    check_preprocessing()
    main_menu()