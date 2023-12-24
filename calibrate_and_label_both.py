# -*- coding: utf-8 -*-

"""
##calibrate stereo cameras
"""

import os
import deeplabcut
from google.colab import drive

gdrive_base = os.path.join('/content/drive/')
drive.mount(gdrive_base)
gdrive_tmp = os.path.join(gdrive_base, "MyDrive/tmp/")
os.makedirs(gdrive_tmp, exist_ok=True)

project = '2nd_stereo_3d_test'
labeler = 'rob_bain'

save_dir = f'{project}-{labeler}-2023-12-23-3d'
conf_path_3d = os.path.join(gdrive_tmp, save_dir, 'config.yaml')

# calibrate=True assumes that you've already viewed and filtered your calibration images
# deeplabcut.calibrate_cameras(conf_path_3d, cbrow=6, cbcol=8, calibrate=True, alpha=0.9)
deeplabcut.check_undistortion(conf_path_3d, cbrow=6, cbcol=8)

# deeplabcut.triangulate(conf_path_3d, '/home/rbain/git/mcn_dlc3d/', videotype=".avi", filterpredictions=True)
# deeplabcut.create_labeled_video_3d(conf_path_3d, ['camera_1.avi', 'camera_2.avi'], videofolder="/home/rbain/git/mcn_dlc3d/", videotype=".avi")

# import os

# try:
  # os.remove('/usr/local/dlclibrary/modelzoo_urls.yaml')
  # os.remove('/usr/local/lib/python3.10/dist-packages/dlclibrary/dlcmodelzoo/modelzoo_urls.yaml')
# except:
#   pass
# !wget http://rkbain.com/modelzoo_urls.yaml -O '/usr/local/dlclibrary/modelzoo_urls.yaml'
# !wget http://rkbain.com/modelzoo_urls.yaml -O '/usr/local/lib/python3.10/dist-packages/dlclibrary/dlcmodelzoo/modelzoo_urls.yaml'

import deeplabcut
# print(deeplabcut.__version__)
# import dlclibrary
# print(dlclibrary.__version__)
import os
import cv2

base_path = os.path.join('/content/drive/')
drive.mount(base_path)
v_path_1 = "MyDrive/McNaughton Lab/dlc3d_clone/12_12_2023_first_stereo/camera_1_1702418161.avi"
v_path_2 = "MyDrive/McNaughton Lab/dlc3d_clone/12_12_2023_first_stereo/camera_3_1702418197.avi"
video_path_1 = os.path.join(base_path, v_path_1)
video_path_2 = os.path.join(base_path, v_path_2)
video_name_1 = os.path.splitext(video_path_1)[0]
video_name_2 = os.path.splitext(video_path_2)[0]

supermodel_name = "superanimal_topviewmouse"
pcutoff = 0.8
videotype = os.path.splitext(video_path_1)[1]
scale_list = []

# @TODO remove hacky code! fixes hangup bug when having not downloaded correctly.
# Hacky solution: Just re-download every time
import shutil;
try:
  shutil.rmtree("/usr/local/lib/python3.10/dist-packages/deeplabcut/pose_estimation_tensorflow/models/pretrained/")
  assert False, "dude, fix me!"
except:
  pass

video_1 = cv2.VideoCapture(video_path_1)
fps = video_1.get(cv2.CAP_PROP_FPS)
print(f'video FPS: {fps}')
frame_count = 0
while(True):
    ret, img = video_1.read()
    if ret == False:
        break
    frame_count += 1
video_1.release() # start over!
video_1 = cv2.VideoCapture(video_path_1)
video_2 = cv2.VideoCapture(video_path_2)
print(f'video frames: {frame_count}')
duration = frame_count / fps
h = int(video_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(video_1.get(cv2.CAP_PROP_FRAME_WIDTH))
short_video_path_1 = "/tmp/short_1.avi"
short_video_path_2 = "/tmp/short_2.avi"
# clean up the previous time it ran
for f in os.listdir("/tmp"):
    if "short" in f:
        os.remove(os.path.join("/tmp", f))
dur_cutoff = 10
if duration > dur_cutoff:
    print(f"video duration too long, making {dur_cutoff} sec trimmed copy")
    vid_frames = 0
    fourcc = 0
    short_video_1 = cv2.VideoWriter(short_video_path_1, fourcc, fps, (w,h))
    short_video_2 = cv2.VideoWriter(short_video_path_2, fourcc, fps, (w,h))
    while(vid_frames < dur_cutoff * fps):
        ret, img_1 = video_1.read()
        ret, img_2 = video_2.read()
        if ret == False:
            print("\nshould not have made it to this line of code!\n")
            break
        vid_frames += 1
        short_video_1.write(img_1)
        short_video_2.write(img_2)
    print(f'Number of frames in shortened video: {vid_frames}')
    video_1.release()
    video_2.release()
    short_video_1.release()
    short_video_2.release()
    video_path_1 = short_video_path_1
    video_path_2 = short_video_path_2
    video_name_1 = os.path.splitext(video_path_1)[0]
    video_name_2 = os.path.splitext(video_path_2)[0]

deeplabcut.video_inference_superanimal(
    [video_path_1, video_path_2],
    supermodel_name,
    videotype=videotype,
    video_adapt=True,
    scale_list=scale_list,
    pcutoff=pcutoff,
)

import h5py
import pandas as pd

predictions_1 = os.path.join(video_name_1 + 'DLC_snapshot-1000.h5')
predictions_2 = os.path.join(video_name_2 + 'DLC_snapshot-1000.h5')
print(predictions_1)
print(predictions_2)

df_1 = pd.read_hdf(predictions_1)
x_1 = []
y_1 = []
prob_1 = []
for i in range(len(df_1)):
    frame = df_1.iloc[i]
    for j in range(0, len(frame), 3):
        x_1.append(frame[j])
        y_1.append(frame[j+1])
        prob_1.append(frame[j+2])

df_2 = pd.read_hdf(predictions_2)
x_2 = []
y_2 = []
prob_2 = []
for i in range(len(df_2)):
    frame = df_2.iloc[i]
    for j in range(0, len(frame), 3):
        x_2.append(frame[j])
        y_2.append(frame[j+1])
        prob_2.append(frame[j+2])

pcutoff = 0.75
ones_to_rm = []
for i, (p1, p2) in enumerate(zip(prob_1, prob_2)):
    if p1 < pcutoff or p2 < pcutoff:
        ones_to_rm.append(i)

num_rm_so_far = 0
for j in ones_to_rm:
    i = j - num_rm_so_far
    del x_1[i], x_2[i]
    del y_1[i], y_2[i]
    del prob_1[i], prob_2[i]
    num_rm_so_far += 1

import cv2
import numpy as np
import matplotlib.pyplot as plt
from deeplabcut.utils import auxiliaryfunctions

path_camera_matrix = os.path.join(gdrive_tmp, save_dir, "camera_matrix/")
path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
stereo_info = stereo_file['camera-1-camera-2']
P1 = stereo_info['P1']
P2 = stereo_info['P2']
F = stereo_info["F"]

triangulate = []

for i in range(len(x_1)):
    cam1_pts = np.array([x_1[i], y_1[i]]).T
    cam2_pts = np.array([x_2[i], y_2[i]]).T

    _3d_pts = cv2.triangulatePoints(P1[:3], P2[:3], cam1_pts, cam2_pts)
    _3d_pts = (_3d_pts / _3d_pts[3])[:-1]
    triangulate.append(_3d_pts)

assert len(triangulate) == len(x_1), "lists no longer parallel :("

triangulate = np.array(triangulate)
colormap = plt.get_cmap("jet")
col = colormap(np.linspace(0, 1, triangulate.shape[0]))
markerSize = 15
markerType = '*'
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(triangulate.shape[0]):
    xs = triangulate[i, 0]
    ys = triangulate[i, 1]
    zs = triangulate[i, 2]
    ax.scatter(xs, ys, zs, c=col[i], marker=markerType, s=markerSize)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
plt.savefig("mouse_3d_kypts.png")
plt.show()

import random

rand_idx = int((random.random() * 100))
# print(triangulate[rand_idx])
print(max(triangulate[:,0]))
print(min(triangulate[:,0]))
print()
print(min(triangulate[:,1]))
print(max(triangulate[:,1]))
print()
print(min(triangulate[:,2]))
print(max(triangulate[:,2]))

