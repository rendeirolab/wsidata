import shutil
from pathlib import Path

root = Path(__file__).parent  # ./docs

target_folders = [
    root / "build",
    root / "api" / "_autogen",
]


if __name__ == "__main__":
    for folder in target_folders:
        if folder.exists():
            shutil.rmtree(folder)
