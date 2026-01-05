#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Sensor Fusion Dashboard - Code Quality Check${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Track overall status
OVERALL_STATUS=0

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Function to print section header
print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# 1. Black Formatting Check
print_section "1/3 - Black Code Formatting Check"
echo "Checking if code is formatted according to Black style guide..."
echo ""

if black --check src/ examples/ run.py train_character_model.py 2>&1 | tail -1; then
    echo -e "${GREEN}✅ Code formatting is correct${NC}"
else
    echo -e "${RED}❌ Code formatting issues found${NC}"
    echo -e "${YELLOW}💡 Run 'black src/ examples/ run.py train_character_model.py' to auto-fix${NC}"
    OVERALL_STATUS=1
fi

# 2. Pylint Analysis
print_section "2/3 - Pylint Code Quality Analysis"
echo "Running Pylint with minimum score threshold of 9.5..."
echo ""

if pylint src/ --fail-under=9.5 --output-format=colorized --score=yes; then
    echo -e "${GREEN}✅ Pylint analysis passed (score >= 9.5)${NC}"
else
    echo -e "${RED}❌ Pylint score is below 9.5 threshold${NC}"
    echo -e "${YELLOW}💡 Review the output above and fix the issues${NC}"
    OVERALL_STATUS=1
fi

# 3. Run Tests
print_section "3/3 - Running Test Suite"
echo "Executing pytest with coverage..."
echo ""

if pytest tests/ -v --tb=short --cov=src --cov-report=term-missing; then
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "${YELLOW}💡 Review the test output above${NC}"
    OVERALL_STATUS=1
fi

# Final Summary
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - Code is ready for commit!${NC}"
else
    echo -e "${RED}❌ SOME CHECKS FAILED - Please fix the issues above${NC}"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $OVERALL_STATUS
