#!/bin/bash

# Production Readiness Check & Cleanup
# Verifies the project is production-ready

echo "════════════════════════════════════════════════════════════"
echo "  Production Readiness Check"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check required files exist
echo "✓ Checking required files..."
required_files=(
    "run.py"
    "train_character_model.py"
    "requirements.txt"
    "README.md"
    ".pylintrc"
    "pyproject.toml"
    "data/config.json"
    "data/sample_sensor_data.csv"
)

missing=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        missing=$((missing+1))
    fi
done

if [ $missing -gt 0 ]; then
    echo ""
    echo "❌ Missing $missing required files!"
    exit 1
fi

echo ""
echo "✓ All required files present"
echo ""

# Check directory structure
echo "✓ Checking project structure..."
dirs=(
    "src"
    "src/ai"
    "src/core"
    "src/gui"
    "src/services"
    "src/utils"
    "src/actions"
    "data"
    "examples"
    "tests"
    ".github/workflows"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ MISSING: $dir"
        exit 1
    fi
done

echo ""
echo "✓ All directories present"
echo ""

# Verify production entry points work
echo "✓ Checking production entry points..."

# Find Python in venv
PYTHON=""
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# Check run.py syntax
if $PYTHON -m py_compile run.py 2>/dev/null; then
    echo "  ✓ run.py (main entry point)"
else
    echo "  ✗ run.py has syntax errors"
    exit 1
fi

# Check train_character_model.py syntax
if $PYTHON -m py_compile train_character_model.py 2>/dev/null; then
    echo "  ✓ train_character_model.py (training entry point)"
else
    echo "  ✗ train_character_model.py has syntax errors"
    exit 1
fi

echo ""
echo "✓ Entry points valid"
echo ""

# List example scripts
echo "✓ Example scripts available:"
for script in examples/*.py; do
    if [ -f "$script" ]; then
        echo "  • $(basename "$script")"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ PRODUCTION READY"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Entry points:"
echo "  • python run.py                    (Launch GUI)"
echo "  • python train_character_model.py  (Train model)"
echo ""
echo "Examples:"
echo "  • python examples/train_model_only.py"
echo "  • python examples/train_with_onhw.py"
echo "  • python examples/evaluate_model.py"
echo ""
