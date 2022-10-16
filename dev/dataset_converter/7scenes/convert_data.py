import cv2
import numpy as np
import torch
from path import Path
import os

from config import Config
from dataset_loader import PreprocessImage, load_image


def process(device, K, poses, image_filenames, depth_filenames):
    scale_rgb = 255.0
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]

    save_data = {}

    preprocessor = PreprocessImage(K=K,
                                old_width=Config.org_image_width,
                                old_height=Config.org_image_height,
                                new_width=Config.test_image_width,
                                new_height=Config.test_image_height,
                                distortion_crop=0,
                                perform_crop=False)

    full_K_torch = torch.from_numpy(preprocessor.get_updated_intrinsics()).float().to(device).unsqueeze(0)

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
        reference_image_torch = torch.from_numpy(np.transpose(reference_image, (2, 0, 1))).float().to(device).unsqueeze(0)
        reference_pose_torch = torch.from_numpy(reference_pose).float().to(device).unsqueeze(0)

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

    return save_data


def prepare(test_dataset_name):
    device = torch.device("cuda")

    scene = test_dataset_name
    scene_folder = Path('%s/%s' % (Config.test_online_scene_path, scene))

    print("Processing for scene:", scene)
    K = np.loadtxt(scene_folder / 'K.txt').astype(np.float32)
    poses = np.fromfile(scene_folder / "poses.txt", dtype=float, sep="\n ").reshape((-1, 4, 4))
    image_filenames = sorted((scene_folder / 'images').files("*.png"))
    depth_filenames = sorted((scene_folder / 'depth').files("*.png"))

    return device, K, poses, image_filenames, depth_filenames


if __name__ == '__main__':
    dataset_names = {}
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]

    save_dir = Path(os.path.dirname(os.path.abspath(__file__))) / 'data'
    os.makedirs(save_dir, exist_ok=True)

    for test_dataset_name in test_dataset_names:
        args = prepare(test_dataset_name)
        print("Processing: %s" % test_dataset_name)
        save_data = process(*args)
        np.savez_compressed(save_dir / test_dataset_name, **save_data)
