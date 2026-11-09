"""Check a local SuSy Dataset copy against the official file list.

The dataset repository publishes susy_dataset.json, the exact file names of
every split and class. Calibration picks its images from the sorted file list
of the train split, so a copy with missing or extra files calibrates on other
images and gives other quantized models.

    uv run python scripts/check_dataset.py                    # train, test and val
    uv run python scripts/check_dataset.py --root ~/datalake/SuSy-Dataset/data --splits train

Exits 1 if any split differs.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

LIST_URL = "https://huggingface.co/datasets/HPAI-BSC/SuSy-Dataset/resolve/main/susy_dataset.json"
# Where the flow reads it: ${paths.data_dir}/SuSy-Dataset/data/<split>.
DEFAULT_ROOT = Path.home() / "datalake" / "SuSy-Dataset" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="folder with train/ val/ test/")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    with urllib.request.urlopen(LIST_URL, timeout=60) as response:
        official = json.load(response)

    ok = True
    for split in args.splits:
        print(f"\n{split} ({args.root / split})")
        print(f"  {'class':20s}{'official':>9s}{'local':>7s}{'missing':>9s}{'extra':>7s}")
        classes = sorted(set(official[split]) | {p.name for p in (args.root / split).glob("*") if p.is_dir()})
        for name in classes:
            expected = set(official[split].get(name, []))
            folder = args.root / split / name
            # Recursive, like the calibration reader and ImageFolder: an extra
            # nesting level (e.g. from unzipping) still counts.
            found = {p.name for p in folder.rglob("*") if p.is_file()} if folder.is_dir() else set()
            missing, extra = expected - found, found - expected
            ok &= not missing and not extra
            print(f"  {name:20s}{len(expected):>9d}{len(found):>7d}{len(missing):>9d}{len(extra):>7d}")
            for label, files in (("missing", missing), ("extra", extra)):
                if files:
                    print(f"    {label}: {', '.join(sorted(files)[:5])}{' ...' if len(files) > 5 else ''}")
    print("\nMatches the official file list." if ok else "\nDiffers from the official file list.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
