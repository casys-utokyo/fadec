import os
from path import Path
import struct

import cv2

def convert_hololens():
    image_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../../dataset/hololens-dataset/000/images/")
    save_dir = os.path.dirname(os.path.abspath(__file__)) / Path("images")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(20):
        print("Processing %05d.png" % (i+3))
        image_file = image_dir / "%05d.png" % (i+3)
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        assert image.shape == (3, 360, 540)

        with open(save_dir / "%05d" % (i+3), 'wb') as f:
            for v in image.reshape(-1):
                f.write(struct.pack('B', v))


if __name__ == '__main__':
    convert_hololens()
