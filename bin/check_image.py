import argparse
from pathlib import Path
from PIL import Image, ImageFilter
import cv2
import numpy as np
from celestial_nav import LostInSpace

# DB_PATH = Path(".cache/t3_5_120_7.npz")
DB_PATH = Path(".cache/t3_40_60_7.npz")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    image_path = Path(args.file)

    print(f"Checking image: {image_path}")

    image = Image.open(image_path)

    # img = np.asarray(image, dtype=np.float32)
    # background = cv2.GaussianBlur(img, (0, 0), 20)
    # img = img - background
    # img = np.maximum(img, 0)
    # white = np.percentile(img, 99.9)
    # img = np.clip(img / white, 0, 1)

    # Image.fromarray(
    #     (img * 65535).astype(np.uint16)
    # ).show()

    los = LostInSpace(DB_PATH)
    solution = los.solve(image, distortion=(-0.25, 0.25))
    print(f"Solution: {solution}")
    if solution:
        solution.visual.show()

    
if __name__ == "__main__":
    main()