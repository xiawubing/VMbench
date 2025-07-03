# -*- coding: utf-8 -*-
"""
red_ellipse_mask.py – Automatically mask the region inside a red ellipse by turning
those pixels black.

Pipeline
--------
1. **Detect red pixels** – Convert the image to HSV and threshold for the two hue
   ranges that represent red.
2. **Locate the ellipse boundary** – From the binary red mask, scan from *top*,
   *bottom*, *left* and *right* until the first red pixel is found.  These four
   intercepts give us the bounding‑box corners.
3. **Build an elliptical mask** – Use the bounding‑box centre/axes to draw a
   filled ellipse in a single‑channel mask.
4. **Apply the mask** – Assign `(0,0,0)` to all pixels where the mask is 255.
5. **Persist** – Save the result as `<original_stem>_mask.jpg` in the same
   directory and print the bounding‑box coordinates.

Usage
-----
```bash
python red_ellipse_mask.py /path/to/your_image.jpg
```

Requires: `opencv-python`, `numpy`.
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_red_bbox(red_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x_left, y_top, x_right, y_bottom) of the first red pixels hit
    by scanning from the four cardinal directions.
    """
    # Scan rows → any red pixel in this row?
    rows_has_red = np.any(red_mask, axis=1)
    cols_has_red = np.any(red_mask, axis=0)

    y_top = int(np.argmax(rows_has_red))  # first True from top
    y_bottom = int(len(rows_has_red) - 1 - np.argmax(rows_has_red[::-1]))
    x_left = int(np.argmax(cols_has_red))  # first True from left
    x_right = int(len(cols_has_red) - 1 - np.argmax(cols_has_red[::-1]))

    if x_left >= x_right or y_top >= y_bottom:
        raise RuntimeError("Failed to detect a valid red ellipse. Check threshold.")

    return x_left, y_top, x_right, y_bottom


def mask_red_ellipse(image_path: str | Path) -> tuple[tuple[int, int, int, int], Path]:
    """Cover the inside of the red ellipse with black pixels.

    Returns
    -------
    bbox : (x_left, y_top, x_right, y_bottom)
    out_path : path to the saved *_mask.jpg file
    """
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    # --- 1. Detect red pixels in HSV space ----------------------------------
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 根据你的RGB范围 (240,0,0) 到 (250,10,10) 转换的HSV阈值
    lower_red = np.array([0, 245, 240])   # 对应RGB (240,0,0)
    upper_red = np.array([0, 255, 250])   # 对应RGB (250,10,10)

    red_mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # --- 2. Bounding‑box via four scanning lines -----------------------------
    x_left, y_top, x_right, y_bottom = find_red_bbox(red_mask)

    # --- 3. Build an elliptical mask based on the bbox -----------------------
    center = ((x_left + x_right) // 2, (y_top + y_bottom) // 2)
    axes = ((x_right - x_left) // 2, (y_bottom - y_top) // 2)
    ellipse_mask = np.zeros_like(red_mask)  # single channel
    cv2.ellipse(ellipse_mask, center, axes, angle=0, startAngle=0, endAngle=360,
                color=255, thickness=-1)

    # --- 4. Apply the mask ----------------------------------------------------
    img_bgr[ellipse_mask != 255] = (0, 0, 0)  # only keep inside ellipse, set outside to black

    # --- 5. Save --------------------------------------------------------------
    out_path = image_path.with_stem(image_path.stem + "_keep_red").with_suffix(".jpg")
    cv2.imwrite(str(out_path), img_bgr)
    return (x_left, y_top, x_right, y_bottom), out_path


def process_all_images_in_folder(folder_path: str | Path):
    """Process all images in the specified folder."""
    folder_path = Path(folder_path)
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f'*{ext}'))
        image_files.extend(folder_path.glob(f'*{ext.upper()}'))

    # 只保留未被处理过的图片
    image_files = [img for img in image_files if not (img.stem.endswith('_mask') or img.stem.endswith('_keep_red'))]

    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process...")
    
    success_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            bbox, saved_path = mask_red_ellipse(img_path)
            success_count += 1
            print(f"✓ {img_path.name}: bbox={bbox}, saved to {saved_path.name}")
        except Exception as e:
            print(f"✗ {img_path.name}: Error - {e}")
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(image_files)} images.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "--batch":
            # 批量处理 red_circle 文件夹
            process_all_images_in_folder("red_circle")
        else:
            # 单个文件处理
            bbox, saved_path = mask_red_ellipse(sys.argv[1])
            print("Bounding‑box (x_left, y_top, x_right, y_bottom):", bbox)
            print("Masked image saved to:", saved_path)
    else:
        print("Usage:")
        print("  Single image: python cover_red.py /path/to/image.jpg")
        print("  Batch process: python cover_red.py --batch")
