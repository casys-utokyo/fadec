import cv2
import numpy as np


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class PreprocessImage:
    def __init__(self, K, old_width, old_height, new_width, new_height, distortion_crop=0, perform_crop=True):
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.new_width = new_width
        self.new_height = new_height
        self.perform_crop = perform_crop

        original_height = np.copy(old_height)
        original_width = np.copy(old_width)

        if self.perform_crop:
            old_height -= 2 * distortion_crop
            old_width -= 2 * distortion_crop

            old_aspect_ratio = float(old_width) / float(old_height)
            new_aspect_ratio = float(new_width) / float(new_height)

            if old_aspect_ratio > new_aspect_ratio:
                # we should crop horizontally to decrease image width
                target_width = old_height * new_aspect_ratio
                self.crop_x = int(np.floor((old_width - target_width) / 2.0)) + distortion_crop
                self.crop_y = distortion_crop
            else:
                # we should crop vertically to decrease image height
                target_height = old_width / new_aspect_ratio
                self.crop_x = distortion_crop
                self.crop_y = int(np.floor((old_height - target_height) / 2.0)) + distortion_crop

            self.cx -= self.crop_x
            self.cy -= self.crop_y
            intermediate_height = original_height - 2 * self.crop_y
            intermediate_width = original_width - 2 * self.crop_x

            factor_x = float(new_width) / float(intermediate_width)
            factor_y = float(new_height) / float(intermediate_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y
        else:
            self.crop_x = 0
            self.crop_y = 0
            factor_x = float(new_width) / float(original_width)
            factor_y = float(new_height) / float(original_height)

            self.fx *= factor_x
            self.fy *= factor_y
            self.cx *= factor_x
            self.cy *= factor_y

    def apply_rgb(self, image, scale_rgb, mean_rgb, std_rgb, normalize_colors=True):
        raw_height, raw_width, _ = image.shape
        cropped_image = image[self.crop_y:raw_height - self.crop_y, self.crop_x:raw_width - self.crop_x, :]
        cropped_image = cv2.resize(cropped_image, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        if normalize_colors:
            cropped_image = cropped_image / scale_rgb
            cropped_image[:, :, 0] = (cropped_image[:, :, 0] - mean_rgb[0]) / std_rgb[0]
            cropped_image[:, :, 1] = (cropped_image[:, :, 1] - mean_rgb[1]) / std_rgb[1]
            cropped_image[:, :, 2] = (cropped_image[:, :, 2] - mean_rgb[2]) / std_rgb[2]
        return cropped_image

    def get_updated_intrinsics(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])
