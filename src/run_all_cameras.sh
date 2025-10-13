#!/bin/bash
# Run auto-optimizer for all cameras using organized input structure

echo "=========================================="
echo "Auto-Optimizing All Cameras"
echo "=========================================="

cd "$(dirname "$0")"

# 1. Original 128px (VISTA)
echo ""
echo "1/3: Original 128px (VISTA)"
python auto_optimize_clearsky.py \
  --image ../input/original_128px_rgb.png \
  --mask ../input/masks/original_128px_mask.png \
  --output ../output/original_128px_auto

# 2. Mobotix
echo ""
echo "2/3: Mobotix"
python auto_optimize_clearsky.py \
  --image ../input/mobotix_rgb.png \
  --mask ../input/masks/mobotix_mask.png \
  --output ../output/mobotix_auto

# 3. Pyranovision
echo ""
echo "3/3: Pyranovision"
python auto_optimize_clearsky.py \
  --image ../input/pyranovision_rgb.png \
  --mask ../input/masks/pyranovision_mask.png \
  --output ../output/pyranovision_auto

echo ""
echo "=========================================="
echo "✅ All cameras processed!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - output/original_128px_auto/"
echo "  - output/mobotix_auto/"
echo "  - output/pyranovision_auto/"
echo ""
echo "View results:"
echo "  - output/*/best_synthetic_clearsky.png"
echo "  - output/*/best_comparison.png"
echo "  - output/*/optimization_results.json"
