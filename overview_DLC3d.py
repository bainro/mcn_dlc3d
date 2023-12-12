#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import deeplabcut

project_name = 'stereo_3d_test'
labeler_name = 'rob_bain'
# conf_path_3d = deeplabcut.create_new_project_3d(project_name, labeler_name, num_cameras=2)

conf_path_3d = "/home/rbain/git/mcn_dlc3d/stereo_3d_test-rob_bain-2023-12-11-3d"
conf_path_3d = os.path.join(conf_path_3d, 'config.yaml')

deeplabcut.calibrate_cameras(conf_path_3d, cbrow=5, cbcol=7, calibrate=True, alpha=0.9)
deeplabcut.check_undistortion(conf_path_3d, cbrow=5, cbcol=7)
