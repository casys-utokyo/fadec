# TAKEN FROM KORNIA LIBRARY

from typing import Optional
import torch
import torch.nn.functional as F

def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(obj)))


def create_meshgrid(
        height: int,
        width: int,
        normalized_coordinates: bool = True,
        device: Optional[torch.device] = torch.device('cpu')) -> torch.Tensor:
    """Generates a coordinate grid for an image.

    When the flag `normalized_coordinates` is set to True, the grid is
    normalized to be in the range [-1,1] to be consistent with the pytorch
    function grid_sample.
    http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample

    Args:
        height (int): the image height (rows).
        width (int): the image width (cols).
        normalized_coordinates (bool): whether to normalize
          coordinates in the range [-1, 1] in order to be consistent with the
          PyTorch function grid_sample.

    Return:
        torch.Tensor: returns a grid tensor with shape :math:`(1, H, W, 2)`.
    """
    # generate coordinates
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coordinates:
        xs = torch.linspace(-1, 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(-1, 1, height, device=device, dtype=torch.float)
    else:
        xs = torch.linspace(0, width - 1, width, device=device, dtype=torch.float)
        ys = torch.linspace(0, height - 1, height, device=device, dtype=torch.float)
    # generate grid by stacking coordinates
    base_grid: torch.Tensor = torch.stack(
        torch.meshgrid([xs, ys], indexing="ij")).transpose(1, 2)  # 2xHxW
    return torch.unsqueeze(base_grid, dim=0).permute(0, 2, 3, 1)  # 1xHxWx2


def unproject_points(
        point_2d: torch.Tensor,
        depth: torch.Tensor,
        camera_matrix: torch.Tensor,
        normalize: bool = False) -> torch.Tensor:
    r"""Unprojects a 2d point in 3d.

    Transform coordinates in the pixel frame to the camera frame.

    Args:
        point2d (torch.Tensor): tensor containing the 2d to be projected to
            world coordinates. The shape of the tensor can be :math:`(*, 2)`.
        depth (torch.Tensor): tensor containing the depth value of each 2d
            points. The tensor shape must be equal to point2d :math:`(*, 1)`.
        camera_matrix (torch.Tensor): tensor containing the intrinsics camera
            matrix. The tensor shape must be Bx4x4.
        normalize (bool, optional): whether to normalize the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position. Default is `False`.

    Returns:
        torch.Tensor: tensor of (x, y, z) world coordinates with shape
        :math:`(*, 3)`.
    """
    if not torch.is_tensor(point_2d):
        raise TypeError("Input point_2d type is not a torch.Tensor. Got {}"
                        .format(type(point_2d)))

    if not torch.is_tensor(depth):
        raise TypeError("Input depth type is not a torch.Tensor. Got {}"
                        .format(type(depth)))

    if not torch.is_tensor(camera_matrix):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}"
                        .format(type(camera_matrix)))

    if not (point_2d.device == depth.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_2d.shape[-1] == 2:
        raise ValueError("Input points_2d must be in the shape of (*, 2)."
                         " Got {}".format(point_2d.shape))

    if not depth.shape[-1] == 1:
        raise ValueError("Input depth must be in the shape of (*, 1)."
                         " Got {}".format(depth.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")
    # projection eq. K_inv * [u v 1]'
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy

    # unpack coordinates
    u_coord: torch.Tensor = point_2d[..., 0]
    v_coord: torch.Tensor = point_2d[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # projective
    x_coord: torch.Tensor = (u_coord - cx) / fx
    y_coord: torch.Tensor = (v_coord - cy) / fy

    xyz: torch.Tensor = torch.stack([x_coord, y_coord], dim=-1)
    xyz = convert_points_to_homogeneous(xyz)

    if normalize:
        xyz = F.normalize(xyz, dim=-1, p=2)

    return xyz * depth


def depth_to_3d(depth: torch.Tensor, camera_matrix: torch.Tensor, normalize_points: bool = False) -> torch.Tensor:
    """Compute a 3d point per pixel given its depth value and the camera intrinsics.

    Args:
        depth (torch.Tensor): image tensor containing a depth value per pixel.
        camera_matrix (torch.Tensor): tensor containing the camera intrinsics.
        normalize_points (bool): whether to normalise the pointcloud. This
            must be set to `True` when the depth is represented as the Euclidean
            ray length from the camera position. Default is `False`.

    Shape:
        - Input: :math:`(B, 1, H, W)` and :math:`(B, 3, 3)`
        - Output: :math:`(B, 3, H, W)`

    Return:
        torch.Tensor: tensor with a 3d point per pixel of the same resolution as the input.

    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"Input depht type is not a torch.Tensor. Got {type(depth)}.")

    if not len(depth.shape) == 4 and depth.shape[-3] == 1:
        raise ValueError(f"Input depth musth have a shape (B, 1, H, W). Got: {depth.shape}")

    if not isinstance(camera_matrix, torch.Tensor):
        raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                        f"Got {type(camera_matrix)}.")

    if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                         f"Got: {camera_matrix.shape}.")

    # create base coordinates grid
    batch_size, _, height, width = depth.shape
    points_2d: torch.Tensor = create_meshgrid(
        height, width, normalized_coordinates=False)  # 1xHxWx2
    points_2d = points_2d.to(depth.device).to(depth.dtype)

    # depth should come in Bx1xHxW
    points_depth: torch.Tensor = depth.permute(0, 2, 3, 1)  # 1xHxWx1

    # project pixels to camera frame
    camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1x3x3
    points_3d: torch.Tensor = unproject_points(
        points_2d, points_depth, camera_matrix_tmp, normalize=normalize_points)  # BxHxWx3

    return points_3d.permute(0, 3, 1, 2)  # Bx3xHxW


def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that converts points from homogeneous to Euclidean space.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))

    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    # we check for points at infinity
    z_vec: torch.Tensor = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask: torch.Tensor = torch.abs(z_vec) > eps
    scale: torch.Tensor = torch.ones_like(z_vec).masked_scatter_(
        mask, torch.tensor(1.0).to(points.device) / z_vec[mask])

    return scale * points[..., :-1]


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    r"""Function that converts points from Euclidean to homogeneous space.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = kornia.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = kornia.transform_points(trans_01, points_1)  # BxNx3
    """
    check_is_tensor(trans_01)
    check_is_tensor(points_1)
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError("Input batch size must be the same for both tensors or 1")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    return points_0


def project_points(
        point_3d: torch.Tensor,
        camera_matrix: torch.Tensor) -> torch.Tensor:
    r"""Projects a 3d point onto the 2d camera plane.

    Args:
        point3d (torch.Tensor): tensor containing the 3d points to be projected
            to the camera plane. The shape of the tensor can be :math:`(*, 3)`.
        camera_matrix (torch.Tensor): tensor containing the intrinsics camera
            matrix. The tensor shape must be :math:`(*, 3, 3)`.

    Returns:
        torch.Tensor: array of (u, v) cam coordinates with shape :math:`(*, 2)`.
    """
    if not torch.is_tensor(point_3d):
        raise TypeError("Input point_3d type is not a torch.Tensor. Got {}"
                        .format(type(point_3d)))

    if not torch.is_tensor(camera_matrix):
        raise TypeError("Input camera_matrix type is not a torch.Tensor. Got {}"
                        .format(type(camera_matrix)))

    if not (point_3d.device == camera_matrix.device):
        raise ValueError("Input tensors must be all in the same device.")

    if not point_3d.shape[-1] == 3:
        raise ValueError("Input points_3d must be in the shape of (*, 3)."
                         " Got {}".format(point_3d.shape))

    if not camera_matrix.shape[-2:] == (3, 3):
        raise ValueError(
            "Input camera_matrix must be in the shape of (*, 3, 3).")

    # projection eq. [u, v, w]' = K * [x y z 1]'
    # u = fx * X / Z + cx
    # v = fy * Y / Z + cy

    # project back using depth dividing in a safe way
    xy_coords: torch.Tensor = convert_points_from_homogeneous(point_3d)
    x_coord: torch.Tensor = xy_coords[..., 0]
    y_coord: torch.Tensor = xy_coords[..., 1]

    # unpack intrinsics
    fx: torch.Tensor = camera_matrix[..., 0, 0]
    fy: torch.Tensor = camera_matrix[..., 1, 1]
    cx: torch.Tensor = camera_matrix[..., 0, 2]
    cy: torch.Tensor = camera_matrix[..., 1, 2]

    # apply intrinsics ans return
    u_coord: torch.Tensor = x_coord * fx + cx
    v_coord: torch.Tensor = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def normalize_pixel_coordinates(
        pixel_coordinates: torch.Tensor,
        height: int,
        width: int,
        eps: float = 1e-8) -> torch.Tensor:
    r"""Normalize pixel coordinates between -1 and 1.

    Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    Args:
        pixel_coordinates (torch.Tensor): the grid with pixel coordinates.
          Shape can be :math:`(*, 2)`.
        width (int): the maximum width in the x-axis.
        height (int): the maximum height in the y-axis.
        eps (float): safe division by zero. (default 1e-8).

    Return:
        torch.Tensor: the normalized pixel coordinates.
    """
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError("Input pixel_coordinates must be of shape (*, 2). "
                         "Got {}".format(pixel_coordinates.shape))
    # compute normalization factor
    hw: torch.Tensor = torch.stack([
        torch.tensor(width), torch.tensor(height)
    ]).to(pixel_coordinates.device).to(pixel_coordinates.dtype)

    factor: torch.Tensor = torch.tensor(2.) / (hw - 1).clamp(eps)

    return factor * pixel_coordinates - 1