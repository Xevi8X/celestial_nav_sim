# find . -type f -iname '*.raw' -exec python3 raw_to_mono16.py '{}' --width 4056 --height 3040 \;

import argparse
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()

    print(f"Processing file: {args.file}")

    raw = np.fromfile(args.file, dtype="<u2")
    stride = raw.size // args.height
    raw = raw.reshape(args.height, stride)[:, :args.width]
    raw = np.flipud(np.fliplr(raw))
    mono16 = cv2.cvtColor(raw, cv2.COLOR_BayerRG2GRAY)
    output = os.path.splitext(args.file)[0] + ".png"

    if not cv2.imwrite(output, mono16):
        raise RuntimeError(f"Failed to save {output}")

    black = np.percentile(mono16, 0.1)
    white = np.percentile(mono16, 99.9)

    normalized = (
        (mono16.astype(np.float32) - black)
        * 65535.0
        / (white - black)
    )
    normalized = np.clip(normalized, 0, 65535).astype(np.uint16)
    normalized_output = os.path.splitext(args.file)[0] + "_normalized.png"
    if not cv2.imwrite(normalized_output, normalized):
        raise RuntimeError(f"Failed to save {normalized_output}")


if __name__ == "__main__":
    main()