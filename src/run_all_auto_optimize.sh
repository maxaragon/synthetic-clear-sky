#!/bin/bash
# Run auto-optimizer for all cameras

echo "=========================================="
echo "Auto-Optimizing All Cameras"
echo "=========================================="

cd "$(dirname "$0")"

# Original 128px (VISTA)
echo ""
echo "1/5: Original 128px (VISTA)"
python auto_optimize_clearsky.py \
  --image /Users/max/Desktop/PhD/WUR/sky-transfer/synthetic-clearsky-rgb/input/original_128px_rgb_masked.png \
  --mask /Users/max/Desktop/PhD/WUR/sky-transfer/images/processed_masks/ground_truth_frame_20_mask_processed.png \
  --output ../output/original_128px_auto

# Mobotix
echo ""
echo "2/5: Mobotix"
python auto_optimize_clearsky.py \
  --image /Users/max/Desktop/PhD/WUR/sky-transfer/synthetic-clearsky-rgb/input/mobotix_rgb.png \
  --mask /Users/max/Desktop/Mobotix_20240613162000_512x512_mask-2.png \
  --output ../output/mobotix_auto

# Pyranovision
echo ""
echo "3/5: Pyranovision"
python auto_optimize_clearsky.py \
  --image /Users/max/Desktop/PhD/WUR/sky-transfer/synthetic-clearsky-rgb/input/pyranovision_rgb.png \
  --mask /Users/max/Desktop/pyranovision_rgb_mask-4_CORRECTED.png \
  --output ../output/pyranovision_auto

# Arizona
echo ""
echo "4/5: Arizona"
python auto_optimize_clearsky.py \
  --image /Users/max/Desktop/2025-02-24_11_14_00_rgb.png \
  --mask /Users/max/Desktop/2025-02-24_11_14_00_rgb_mask_224x224.png \
  --output ../output/arizona_auto

# PSL
echo ""
echo "5/5: PSL"
python auto_optimize_clearsky.py \
  --image /Users/max/Downloads/2025-10-12T12-00-00+02-00_rgb.webp \
  --mask /Users/max/Downloads/2025-10-12T12-00-00+02-00_rgb_mask-2.png \
  --output ../output/psl_auto

echo ""
echo "=========================================="
echo "✅ All cameras processed!"
echo "=========================================="

