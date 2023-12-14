#!/usr/bin/env python3
import os
import deeplabcut

project_name = 'stereo_3d_test'
labeler_name = 'rob_bain'
# conf_path_3d = deeplabcut.create_new_project_3d(project_name, labeler_name, num_cameras=2)

conf_path_3d = "/home/rbain/git/mcn_dlc3d/stereo_3d_test-rob_bain-2023-12-11-3d"
conf_path_3d = os.path.join(conf_path_3d, 'config.yaml')

# deeplabcut.calibrate_cameras(conf_path_3d, cbrow=6, cbcol=8, calibrate=True, alpha=0.9)
# deeplabcut.check_undistortion(conf_path_3d, cbrow=6, cbcol=8)

# assert False, "change the yaml config file!"

deeplabcut.triangulate(conf_path_3d, '/home/rbain/git/mcn_dlc3d/', videotype=".avi", filterpredictions=True)

deeplabcut.create_labeled_video_3d(conf_path_3d, ['camera_1.avi', 'camera_2.avi'], videofolder="/home/rbain/git/mcn_dlc3d/", videotype=".avi")
