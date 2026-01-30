from pathlib import Path

# Project root (current directory)
PROJECT_ROOT = Path.cwd()

# Folder structure
FOLDERS = [
    "src",
    "src/data_ingestion",
    "src/data_preprocessing",
    "src/feature_engineering",
    "src/model_training",
    "src/model_evaluation",
    "src/utils",
    "configs",
    "artifacts",
]

# __init__.py locations
INIT_FILES = [
    "src/__init__.py",
    "src/data_ingestion/__init__.py",
    "src/data_preprocessing/__init__.py",
    "src/feature_engineering/__init__.py",
    "src/model_training/__init__.py",
    "src/model_evaluation/__init__.py",
    "src/utils/__init__.py",
]

def create_folders():
    for folder in FOLDERS:
        path = PROJECT_ROOT / folder
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created folder: {path}")

def create_init_files():
    for file in INIT_FILES:
        path = PROJECT_ROOT / file
        path.touch(exist_ok=True)
        print(f"📄 Created file: {path}")

if __name__ == "__main__":
    print("🚀 Setting up MLOps project structure...\n")
    create_folders()
    create_init_files()
    print("\n✅ Project structure created successfully!")