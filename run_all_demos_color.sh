#!/bin/bash
# Run RGB color-tuned optimizer on all demo inputs

cd /Users/max/Desktop/PhD/WUR/synthetic-clear-sky

for i in 1 2 3 4 5; do
    echo "=========================================="
    echo "Processing Demo $i"
    echo "=========================================="
    
    python src/synthetic_clearsky_optimizer-color.py \
        --image demo/inputs/demo${i}_image.* \
        --mask demo/inputs/demo${i}_mask.png \
        --output demo/outputs_color/demo${i}
    
    echo ""
done

echo "=========================================="
echo "All demos complete!"
echo "=========================================="

