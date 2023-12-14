#!/usr/bin/env python3
# @author: robert bain

import os
import shutil

os.chdir("cams_1_and_3_calib_images/")
files = os.listdir()
#print(files)
#assert False, "hmmm...."
# do the camera-1-*.jpg first
for old_f in files:
    if old_f.startswith("camera-1-"):
        new_f = old_f.replace("camera-1-", "camera-2-")
        shutil.copyfile(old_f, new_f)
        os.remove(old_f)
# loop over again, moveing camera-0-*.jpg to camera-1-*.jpg
for old_f in files:
    if old_f.startswith("camera-0-"):
        new_f = old_f.replace("camera-0-", "camera-1-")
        shutil.copyfile(old_f, new_f)
        os.remove(old_f)
