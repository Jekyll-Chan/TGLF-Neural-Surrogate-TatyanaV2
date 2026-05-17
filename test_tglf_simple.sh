#!/bin/bash
# Simple TGLF test script

GACODE_ROOT="/home/jekyllchan/gacode"
TGLF_BIN="$GACODE_ROOT/tglf/bin/tglf"
TEST_DIR="/tmp/tglf_test_$$"

# Setup gacode environment
export GACODE_ROOT
export PATH="$GACODE_ROOT/tglf/bin:$GACODE_ROOT/shared/bin:$PATH"

mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create a minimal input.tglf first~
cat > input.tglf << 'EOF'
GEOMETRY_FLAG=1
SAT_RULE=3
UNITS='CGYRO'
NKY=1
NS=2
ZS_1=-1
MASS_1=2.7240e-4
AS_1=1.0
ZS_2=1
MASS_2=1.0
AS_2=1.0
KYMIN=0.12715
RMIN_LOC=0.275656
SHAT=0.328106
Q_LOC=1.7683
EOF

echo "Test directory: $TEST_DIR"
echo "GACODE_ROOT: $GACODE_ROOT"
echo "Running TGLF..."
echo ""

"$TGLF_BIN" -e .

echo ""
echo "=== RESULTS ==="
if [ -f out.tglf.eigenvalue_spectrum ]; then
    echo "✓ TGLF completed successfully:)"
    echo ""
    echo "Eigenvalue spectrum:"
    cat out.tglf.eigenvalue_spectrum
else
    echo "✗ No output files generated :("
fi

echo ""
echo "Files created:"
ls -lh out.tglf.* 2>/dev/null || echo "No output files"
