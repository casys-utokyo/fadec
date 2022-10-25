import os
from path import Path
import struct
import shutil

import cv2

def convert_hololens():
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    scene_dir = base_dir / "../../dataset/hololens-dataset/000"
    image_dir = scene_dir / "images"
    save_dir = base_dir / "images_hololens"
    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(scene_dir / "K.txt", save_dir)
    shutil.copy(scene_dir / "poses.txt", save_dir)

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
