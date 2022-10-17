import cv2
import numpy as np
import torch
from path import Path
import struct
import os
import shutil

from config import Config
from dataset_loader import PreprocessImage, load_image


def process_binary(test_dataset_name, images_dir):
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    scene_folder = base_dir / Config.test_online_scene_path / test_dataset_name
    image_dir = images_dir / test_dataset_name
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(scene_folder / "K.txt", image_dir)
    shutil.copy(scene_folder / "poses.txt", image_dir)

    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    for image_file in image_filenames:
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        assert image.shape == (3, 256, 320), image.shape

        filename = image_file.split('/')[-1][:-4]
        with open(image_dir / filename, 'wb') as f:
            for v in image.reshape(-1):
                f.write(struct.pack('B', v))


def process_npz(test_dataset_name, data_dir):
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    scene_folder = base_dir / Config.test_online_scene_path / test_dataset_name

    scale_rgb = 255.0
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    save_data = {}

    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

    preprocessor = PreprocessImage(K=K,
                                   old_width=Config.org_image_width,
                                   old_height=Config.org_image_height,
                                   new_width=Config.test_image_width,
                                   new_height=Config.test_image_height,
                                   distortion_crop=0,
                                   perform_crop=False)

    full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().unsqueeze(0)

    half_K_torch = full_K_torch.clone().cuda()
    half_K_torch[:, 0:2, :] = half_K_torch[:, 0:2, :] / 2.0

    lstm_K_bottom = full_K_torch.clone().cuda()
    lstm_K_bottom[:, 0:2, :] = lstm_K_bottom[:, 0:2, :] / 32.0

    save_data["full_K"] = full_K_torch.cpu().detach().numpy().copy()
    save_data["half_K"] = half_K_torch.cpu().detach().numpy().copy()
    save_data["lstm_K"] = lstm_K_bottom.cpu().detach().numpy().copy()

    for i in range(len(poses)):
        reference_pose = poses[i]
        reference_image = load_image(image_filenames[i])

        reference_image = preprocessor.apply_rgb(image=reference_image, scale_rgb=scale_rgb, mean_rgb=mean_rgb, std_rgb=std_rgb)
        reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().unsqueeze(0)
        reference_pose_torch = torch.from_numpy(reference_pose).float().unsqueeze(0)

        ground_truth = cv2.imread(depth_filenames[i], -1).astype(float) / 10000.0
        ground_truth = cv2.resize(ground_truth, (Config.test_image_width, Config.test_image_height), interpolation=cv2.INTER_NEAREST)

        if i == 0:
            save_data["reference_image"] = reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]
            save_data["reference_pose"] = reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]
            save_data["ground_truth"] = ground_truth[np.newaxis]
        else:
            save_data["reference_image"] = np.vstack([save_data["reference_image"], reference_image_torch.cpu().detach().numpy().copy()[np.newaxis]])
            save_data["reference_pose"] = np.vstack([save_data["reference_pose"], reference_pose_torch.cpu().detach().numpy().copy()[np.newaxis]])
            save_data["ground_truth"] = np.vstack([save_data["ground_truth"], ground_truth[np.newaxis]])

    np.savez_compressed(data_dir / test_dataset_name, **save_data)


if __name__ == '__main__':
    dataset_names = {}
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    print("Processing images_7scenes...")
    images_dir = base_dir / 'images_7scenes'
    if os.path.isdir(images_dir):
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    for test_dataset_name in test_dataset_names:
        print("\tProcessing:", test_dataset_name)
        process_binary(test_dataset_name, images_dir)
        print("\t\tFinished:", test_dataset_name)


    print("Processing data_7scenes...")
    data_dir = base_dir / 'data_7scenes'
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for test_dataset_name in test_dataset_names:
        print("\tProcessing:", test_dataset_name)
        save_data = process_npz(test_dataset_name, data_dir)
        print("\t\tFinished:", test_dataset_name)

