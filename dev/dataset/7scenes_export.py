import os
import shutil

import cv2
import numpy as np
from path import Path

from config import Config
from dataset_loader import PreprocessImage, load_image
from keyframe_buffer2 import KeyframeBuffer


def predict(input_folder, output_folder):
    scene = input_folder.split("/")[-2]
    seq = input_folder.split("/")[-1]
    test_dataset_name = scene + "-" + seq

    keyframe_buffer = KeyframeBuffer(buffer_size=Config.test_keyframe_buffer_size,
                                     keyframe_pose_distance=Config.test_keyframe_pose_distance,
                                     optimal_t_score=Config.test_optimal_t_measure,
                                     optimal_R_score=Config.test_optimal_R_measure,
                                     store_return_indices=False)

    K = np.array([[525.0, 0.0, 320.0],
                  [0.0, 525.0, 240.0],
                  [0.0, 0.0, 1.0]]).astype(np.float32)
    pose_filenames = sorted(input_folder.files("*pose.txt"))
    image_filenames = sorted(input_folder.files("*color.png"))
    depth_filenames = sorted(input_folder.files("*depth.png"))

    poses = []
    for pose_filename in pose_filenames:
        pose = np.loadtxt(pose_filename)
        poses.append(pose.ravel().tolist())
    poses = np.array(poses).reshape(-1, 4, 4)

    save_dir = output_folder / test_dataset_name
    os.makedirs(save_dir / "images", exist_ok=True)
    os.makedirs(save_dir / "depth", exist_ok=True)

    preprocessor = PreprocessImage(K=K,
                                   old_width=Config.org_image_width,
                                   old_height=Config.org_image_height,
                                   new_width=Config.test_image_width,
                                   new_height=Config.test_image_height,
                                   distortion_crop=0,
                                   perform_crop=True)

    with open(save_dir / "K.txt", 'w') as fout:
        for k in preprocessor.get_updated_intrinsics():
            fout.write("%.18e %.18e %.18e\n" % tuple(k))

    for i in range(len(poses)):
        pose = poses[i]

        # POLL THE KEYFRAME BUFFER
        response = keyframe_buffer.try_new_keyframe(pose)
        if response == 2 or response == 4 or response == 5:
            continue
        elif response == 3:
            continue
        image = load_image(image_filenames[i])
        keyframe_buffer.add_new_keyframe(pose, image)

        pose = pose.reshape(16)
        with open(save_dir / "poses.txt", 'a') as fout:
            for j in range(15):
                fout.write("%.18e " % pose[j])
            fout.write("%.18e\n" % pose[15])

        raw_height, raw_width, _ = image.shape
        image = image[preprocessor.crop_y:raw_height - preprocessor.crop_y, preprocessor.crop_x:raw_width - preprocessor.crop_x, :]
        image = cv2.resize(image, (Config.test_image_width, Config.test_image_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir / ("images/%06d.png" % i), image.astype(np.uint8))

        depth = np.round(cv2.imread(depth_filenames[i], -1)).astype(np.uint16)
        raw_height, raw_width = depth.shape
        depth = depth[preprocessor.crop_y:raw_height - preprocessor.crop_y, preprocessor.crop_x:raw_width - preprocessor.crop_x]
        depth = depth.astype(float) * 10.0
        depth = cv2.resize(depth, (Config.test_image_width, Config.test_image_height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_dir / ("depth/%06d.png" % i), depth.astype(np.uint16))


if __name__ == '__main__':
    input_folder = Path("/home/share/dataset/7scenes") # change input_folder path to your environment
    output_folder = Path(os.path.dirname(os.path.abspath(__file__))) / "7scenes"

    input_folders = [
        input_folder / "redkitchen/seq-01",
        input_folder / "redkitchen/seq-07",
        input_folder / "chess/seq-01",
        input_folder / "chess/seq-02",
        input_folder / "fire/seq-01",
        input_folder / "fire/seq-02",
        input_folder / "office/seq-01",
        input_folder / "office/seq-03"]

    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    for f in input_folders:
        print("Processing:", f)
        predict(f, output_folder)
        print("\tFinished:", f)
