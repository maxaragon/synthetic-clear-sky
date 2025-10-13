#!/bin/bash
# Run synthetic clear-sky generation for all cameras using proven per-camera scripts
# These scripts use optimized projection/method combinations for each camera

set -e  # Exit on error

cd "$(dirname "$0")"

echo "========================================================================"
echo "       SYNTHETIC CLEAR-SKY GENERATION - ALL CAMERAS"
echo "========================================================================"
echo ""
echo "Using per-camera optimized scripts:"
echo "  - Original 128px: K-tan K=1.4, Per-channel RGB"
echo "  - Mobotix:        K-tan K=1.6, Y-based (stereo)"
echo "  - Pyranovision:   K-tan K=1.2, Constrained_B"
echo "  - PSL:            K-tan K=1.6, Per-channel RGB"
echo ""
echo "========================================================================"
echo ""

# Original 128px
echo "📷 CAMERA 1/4: Original 128px (VISTA)"
echo "----------------------------------------"
cd src
python fit_original_clearsky.py && python generate_original_synthetic.py
cd ..
echo "✅ Original 128px complete"
echo ""

# Mobotix
echo "📷 CAMERA 2/4: Mobotix"
echo "----------------------------------------"
cd src
python fit_mobotix_clearsky.py && python generate_mobotix_synthetic.py
cd ..
echo "✅ Mobotix complete"
echo ""

# Pyranovision
echo "📷 CAMERA 3/4: Pyranovision"
echo "----------------------------------------"
cd src
python fit_pyranovision_clearsky.py && python generate_pyranovision_synthetic.py
cd ..
echo "✅ Pyranovision complete"
echo ""

# PSL (if fit script exists, otherwise skip)
if [ -f "src/fit_psl_clearsky.py" ]; then
    echo "📷 CAMERA 4/4: PSL"
    echo "----------------------------------------"
    cd src
    python fit_psl_clearsky.py && python generate_psl_synthetic.py
    cd ..
    echo "✅ PSL complete"
else
    echo "📷 CAMERA 4/4: PSL"
    echo "----------------------------------------"
    echo "⚠️  PSL per-camera script not yet created"
    echo "   (Use auto_optimize_clearsky.py for PSL)"
fi
echo ""

echo "========================================================================"
echo "                    ✅ ALL CAMERAS COMPLETE!"
echo "========================================================================"
echo ""
echo "📁 Outputs saved in:"
echo "   output/original_128px/synthetic_clearsky.png"
echo "   output/mobotix/synthetic_clearsky.png"
echo "   output/pyranovision/synthetic_clearsky.png"
echo "   output/psl/synthetic_clearsky.png (if available)"
echo ""
echo "View results:"
echo "   open output/*/synthetic_clearsky.png"
echo ""

