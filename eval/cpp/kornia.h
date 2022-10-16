#pragma once

template <int height, int width>
void create_meshgrid(float meshgrid[height][width][2]) {
    // Generates a coordinate grid for an image.
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        meshgrid[i][j][0] = j;
        meshgrid[i][j][1] = i;
    }
}


template <int height, int width, int channels>
void convert_points_from_homogeneous(const float points[height][width][channels],
                                     float euc_points[height][width][channels-1]) {

    // Function that converts points from homogeneous to Euclidean space.
    const float eps = 1e-8;

    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        const float z = points[i][j][channels-1];
        if (abs(z) > eps) {
            for (int k = 0; k < channels-1; k++) euc_points[i][j][k] = points[i][j][k] / z;
        } else {
            for (int k = 0; k < channels-1; k++) euc_points[i][j][k] = points[i][j][k];
        }
    }
}


template <int height, int width, int channels>
void convert_points_to_homogeneous(const float xyz[height][width][channels],
                                   float padded_xyz[height][width][channels+1]) {
    // Function that converts points from Euclidean to homogeneous space.
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        for (int k = 0; k < channels; k++) padded_xyz[i][j][k] = xyz[i][j][k];
        padded_xyz[i][j][channels] = 1;
    }
}


template <int height, int width>
void project_points(const float points_3d[height][width][3],
                    const float camera_matrix[3][3],
                    float points_2d[height][width][2]) {

    // Projects a 3d point onto the 2d camera plane.

    // project back using depth dividing in a safe way
    float xy_coords[height][width][2];
    convert_points_from_homogeneous<height, width, 3>(points_3d, xy_coords);

    float x_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) x_coord[i][j] = xy_coords[i][j][0];

    float y_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) y_coord[i][j] = xy_coords[i][j][1];

    // unpack intrinsics
    const float fx = camera_matrix[0][0];
    const float fy = camera_matrix[1][1];
    const float cx = camera_matrix[0][2];
    const float cy = camera_matrix[1][2];

    // apply intrinsics ans return
    float u_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) u_coord[i][j] = x_coord[i][j] * fx + cx;

    float v_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) v_coord[i][j] = y_coord[i][j] * fy + cy;

    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        points_2d[i][j][0] = u_coord[i][j];
        points_2d[i][j][1] = v_coord[i][j];
    }
}


template <int height, int width>
void unproject_points(const float points_2d[height][width][2],
                      const float depth[height][width],
                      const float camera_matrix[3][3],
                      float points_3d[height][width][3]) {

    // Unprojects a 2d point in 3d.
    // Transform coordinates in the pixel frame to the camera frame.

    // unpack coordinates
    float u_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) u_coord[i][j] = points_2d[i][j][0];

    float v_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) v_coord[i][j] = points_2d[i][j][1];

    // unpack intrinsics
    const float fx = camera_matrix[0][0];
    const float fy = camera_matrix[1][1];
    const float cx = camera_matrix[0][2];
    const float cy = camera_matrix[1][2];

    // projective
    float x_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) x_coord[i][j] = (u_coord[i][j] - cx) / fx;

    float y_coord[height][width];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) y_coord[i][j] = (v_coord[i][j] - cy) / fy;

    float xyz[height][width][2];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        xyz[i][j][0] = x_coord[i][j];
        xyz[i][j][1] = y_coord[i][j];
    }

    float padded_xyz[height][width][3];
    convert_points_to_homogeneous<height, width, 2>(xyz, padded_xyz);

    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) for (int k = 0; k < 3; k++)
        points_3d[i][j][k] = padded_xyz[i][j][k] * depth[i][j];

}


template <int height, int width>
void depth_to_3d(const float depth[height][width],
                 const float camera_matrix[3][3],
                 float points_3d[height][width][3]) {
    // Compute a 3d point per pixel given its depth value and the camera intrinsics.

    // create base coordinates grid
    float points_2d[height][width][2];
    create_meshgrid<height, width>(points_2d);

    // project pixels to camera frame
    unproject_points<height, width>(points_2d, depth, camera_matrix, points_3d);
}


template <int height, int width>
void transform_points(const Matrix4f trans_01,
                      const float points_1[height][width][3],
                      float points_0[height][width][3]) {
    // Function that applies transformations to a set of points.

    // to homogeneous
    float points_1_h[height][width][4];
    convert_points_to_homogeneous<height, width, 3>(points_1, points_1_h);

    // transform coordinates
    float points_0_h[height][width][4];
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        Vector4f point_1_h;
        for (int k = 0; k < 4; k++) point_1_h(k) = points_1_h[i][j][k];
        Vector4f point_0_h = trans_01 * point_1_h;
        for (int k = 0; k < 4; k++) points_0_h[i][j][k] = point_0_h(k);
    }

    // to euclidean
    convert_points_from_homogeneous<height, width, 4>(points_0_h, points_0);
}


template <int height, int width>
void normalize_pixel_coordinates(const float pixel_coordinates[height][width][2],
                                 float normed_pixel_coordinates[height][width][2]) {


    // Normalize pixel coordinates between -1 and 1.
    // Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1).

    const float eps = 1e-8;

    // compute normalization factor
    const float factor_w = 2.0 / max((float) width - 1, eps);
    const float factor_h = 2.0 / max((float) height - 1, eps);

    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        normed_pixel_coordinates[i][j][0] = factor_w * pixel_coordinates[i][j][0] - 1; // maybe vice versa
        normed_pixel_coordinates[i][j][1] = factor_h * pixel_coordinates[i][j][1] - 1;
    }
}
