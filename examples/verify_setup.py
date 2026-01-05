#!/usr/bin/env python3
"""
Setup and Verification Script
Checks environment, installs dependencies, and verifies components
"""

import sys
import os
import subprocess
import importlib


def check_python_version():
    """Check Python version"""
    print("=" * 70)
    print("PYTHON VERSION CHECK")
    print("=" * 70)

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  ERROR: Python 3.8+ required")
        return False

    print("✓ Python version OK\n")
    return True


def check_dependencies():
    """Check all required dependencies"""
    print("=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)

    dependencies = {
        "numpy": "NumPy (numerical computing)",
        "scipy": "SciPy (scientific computing)",
        "sklearn": "Scikit-learn (ML preprocessing)",
        "matplotlib": "Matplotlib (plotting)",
        "serial": "PySerial (serial communication)",
        "tensorflow": "TensorFlow (deep learning)",
        "keras": "Keras (high-level NN API)",
        "h5py": "H5PY (HDF5 file format)",
        "pandas": "Pandas (data manipulation)",
    }

    missing = []
    installed = []

    for module, name in dependencies.items():
        try:
            lib = importlib.import_module(module)
            version = getattr(lib, "__version__", "unknown")
            installed.append((name, version))
            print(f"✓ {name:40s} v{version}")
        except ImportError:
            missing.append((module, name))
            print(f"✗ {name:40s} NOT INSTALLED")

    print(f"\n{len(installed)}/{len(dependencies)} dependencies installed")

    if missing:
        print(f"\nMissing: {', '.join([name for _, name in missing])}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

    print()
    return True


def check_directory_structure():
    """Check if directory structure is set up correctly"""
    print("=" * 70)
    print("DIRECTORY STRUCTURE CHECK")
    print("=" * 70)

    required_dirs = [
        "src/core",
        "src/gui",
        "src/actions",
        "src/ai/character_recognition",
        "src/ai/utils",
        "src/ai/models",
        "src/utils",
        "data",
        "docs",
    ]

    required_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/gui/__init__.py",
        "src/actions/__init__.py",
        "src/ai/__init__.py",
        "src/ai/character_recognition/__init__.py",
        "src/ai/utils/__init__.py",
        "src/utils/__init__.py",
        "requirements.txt",
        "run.py",
    ]

    all_ok = True

    print("Checking directories:")
    for dir_path in required_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.isdir(full_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} MISSING")
            all_ok = False

    print("\nChecking files:")
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.isfile(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} MISSING")
            all_ok = False

    print()
    return all_ok


def check_module_imports():
    """Check if key modules can be imported"""
    print("=" * 70)
    print("MODULE IMPORT CHECK")
    print("=" * 70)

    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    modules = {
        "core.data_buffers": "Sensor Buffers",
        "core.sensor_parser": "Sensor Parser",
        "core.data_reader": "Data Reader",
        "actions.action_detector": "Action Detector",
        "ai.character_recognition.model": "Character Recognition Model",
        "ai.character_recognition.preprocessor": "Data Preprocessor",
        "ai.character_recognition.trainer": "Model Trainer",
        "ai.utils.data_utils": "Dataset Loader",
    }

    all_ok = True

    for module_path, name in modules.items():
        try:
            mod = importlib.import_module(module_path)
            print(f"✓ {name:40s} OK")
        except Exception as e:
            print(f"✗ {name:40s} ERROR: {str(e)[:40]}")
            all_ok = False

    print()
    return all_ok


def check_model_availability():
    """Check if trained models are available"""
    print("=" * 70)
    print("MODEL AVAILABILITY CHECK")
    print("=" * 70)

    model_path = os.path.join(os.path.dirname(__file__), "src/ai/models/character_recognition_104hz.h5")

    if os.path.isfile(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Pre-trained model found")
        print(f"  Path: src/ai/models/character_recognition_104hz.h5")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print(f"ℹ No pre-trained model found")
        print(f"  Train with: python train_character_model.py")

    print()
    return True


def print_system_info():
    """Print system information"""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    import platform

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")

    print()


def print_next_steps():
    """Print recommended next steps"""
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    print(
        """
1. RUN THE GUI:
   python run.py

2. TRAIN CHARACTER RECOGNITION MODEL:
   python train_character_model.py --dataset synthetic --epochs 50

3. READ DOCUMENTATION:
   - README_AI_INTEGRATION.md: Full guide
   - docs/ARCHITECTURE.md: System architecture
   - docs/USAGE.md: Usage examples

4. VERIFY SENSOR CONNECTION:
   - Connect sensor via USB
   - Select appropriate COM port in GUI
   - Set baudrate to 115200

5. COLLECT TRAINING DATA:
   - Use GUI to record writing samples
   - Label each sample with character (A-Z)
   - Train model with collected data
"""
    )


def main():
    """Run all checks"""
    print("\n")
    print_system_info()

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Module Imports", check_module_imports),
        ("Model Availability", check_model_availability),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"Error during {name} check: {e}\n")
            results.append((name, False))

    # Summary
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")

    all_pass = all(result for _, result in results)

    if all_pass:
        print("\n✓ All checks passed! System ready for use.")
    else:
        print("\n⚠️  Some checks failed. Please review above.")

    print_next_steps()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
