class Config:
    # training settings
    train_min_depth = 0.25
    train_max_depth = 20.0
    train_n_depth_levels = 64

    # test settings
    org_image_width = 540
    org_image_height = 360
    test_image_width = 96
    test_image_height = 64

    test_distortion_crop = 0
    test_perform_crop = False
    test_visualize = False
    test_n_measurement_frames = 2
    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

    # SET THESE: TRAINING FOLDER LOCATIONS
    fusionnet_test_weights = "../params/org_weights"

    # SET THESE: TESTING FOLDER LOCATIONS
    # for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
    test_online_scene_path = "../dataset/hololens-dataset/000"
