#!/bin/bash
# Package Temporal-ZwZ for transfer to another machine.
set -euo pipefail

PROJECT_DIR="/cpfs01/user/chenzhaoyu.czy/code/Temporal-ZwZ"
OUTPUT_DIR="/cpfs01/user/chenzhaoyu.czy/code"
ARCHIVE_NAME="Temporal-ZwZ.tar.gz"

echo "Packaging Temporal-ZwZ..."
echo "  Source: $PROJECT_DIR"
echo "  Output: $OUTPUT_DIR/$ARCHIVE_NAME"

tar -czf "$OUTPUT_DIR/$ARCHIVE_NAME" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='results/' \
    --exclude='.git/' \
    --exclude='*.egg-info' \
    -C "$OUTPUT_DIR" \
    Temporal-ZwZ/

SIZE=$(du -h "$OUTPUT_DIR/$ARCHIVE_NAME" | cut -f1)
echo ""
echo "Done! Archive: $OUTPUT_DIR/$ARCHIVE_NAME ($SIZE)"
echo ""
echo "Transfer to new machine:"
echo "  scp $OUTPUT_DIR/$ARCHIVE_NAME user@remote:/path/"
echo ""
echo "On the new machine:"
echo "  tar -xzf $ARCHIVE_NAME"
echo "  cd Temporal-ZwZ"
echo "  pip install decord opencv-python-headless Pillow numpy pandas pyarrow datasets tqdm"
echo "  # See README.md for full instructions"
