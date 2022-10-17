import os
import shutil

import cv2
import numpy as np
from path import Path


def process_color(input_directories, output_folder):
    for input_directory in input_directories:
        K = np.array([[525.0, 0.0, 320.0],
                    [0.0, 525.0, 240.0],
                    [0.0, 0.0, 1.0]])
        print("\tProcessing: ", input_directory)
        image_filenames = sorted(input_directory.files("*color.png"))
        pose_filenames = sorted(input_directory.files("*pose.txt"))

        poses = []
        for pose_filename in pose_filenames:
            pose = np.loadtxt(pose_filename)
            poses.append(pose)

        scene = input_directory.split("/")[-2]
        seq = input_directory.split("/")[-1]
        current_output_dir = output_folder / (scene + "-" + seq)
        os.makedirs(current_output_dir / "images", exist_ok=True)

        output_poses = []
        for current_index in range(len(image_filenames)):
            image = cv2.imread(image_filenames[current_index])
            output_poses.append(poses[current_index].ravel().tolist())
            cv2.imwrite(current_output_dir / "images/{}.png".format(str(current_index).zfill(6)), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        output_poses = np.array(output_poses)
        np.savetxt(current_output_dir / "poses.txt", output_poses)
        np.savetxt(current_output_dir / "K.txt", K)

        print("\t\tFinished: ", scene + "-" + seq)


def process_depth(input_directories, output_folder):
    for input_directory in input_directories:
        print("\tProcessing: ", input_directory)
        scene = input_directory.split("/")[-2]
        seq = input_directory.split("/")[-1]

        depth_files = sorted(input_directory.files("*depth.png"))
        scene_output_folder = output_folder / (scene + "-" + seq) / 'depth'
        os.makedirs(scene_output_folder, exist_ok=True)

        for index, file in enumerate(depth_files):
            depth = cv2.imread(file, -1)
            depth_uint = np.round(depth).astype(np.uint16)
            save_filename = scene_output_folder / (str(index).zfill(6) + ".png")
            cv2.imwrite(save_filename, depth_uint, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        print("\t\tFinished: ", scene + "-" + seq)


if __name__ == '__main__':
    input_folder = Path("/home/share/dataset/7scenes") # change input_folder path to your environment
    output_folder = Path(os.path.dirname(os.path.abspath(__file__))) / "7scenes"

    input_directories = [
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

    print("Processing color images...")
    process_color(input_directories, output_folder)
    print("Color images finished!\n")

    print("Processing depth maps...")
    process_depth(input_directories, output_folder)
    print("depth maps finished!")
