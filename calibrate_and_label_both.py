import os
import cv2
import deeplabcut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deeplabcut.utils import auxiliaryfunctions

os.chdir("/home/rbain/git/mcn_dlc3d")
_tmp = os.path.join("./", "tmp/")
os.makedirs(_tmp, exist_ok=True)

project = '2nd_stereo_3d_test'
labeler = 'rob_bain'

save_dir = f'{project}-{labeler}-2023-12-25-3d'
conf_path_3d = os.path.join(_tmp, save_dir, 'config.yaml')

### for creating a new project 
#@TODO make this hands-free, not hard-coded...
# deeplabcut.create_new_project_3d(project, labeler, num_cameras=2)

# calibrate=True assumes that you've already viewed and filtered your calibration images
# deeplabcut.calibrate_cameras(conf_path_3d, cbrow=6, cbcol=8, calibrate=True, alpha=0.97)
# deeplabcut.check_undistortion(conf_path_3d, cbrow=6, cbcol=8)

incorrect_yaml = '/home/rbain/anaconda3/envs/dlc3d/lib/python3.9/site-packages/dlclibrary/dlcmodelzoo/modelzoo_urls.yaml'
try:
    os.remove(incorrect_yaml)
except:
    os.system(f"wget http://rkbain.com/modelzoo_urls.yaml -O '{incorrect_yaml}'")

v_path_1 = "12_12_2023_first_stereo/camera_1_1702418161.avi"
v_path_2 = "12_12_2023_first_stereo/camera_3_1702418197.avi"
video_path_1 = os.path.join("./", v_path_1)
video_path_2 = os.path.join("./", v_path_2)
video_name_1 = os.path.splitext(video_path_1)[0]
video_name_2 = os.path.splitext(video_path_2)[0]

supermodel_name = "superanimal_topviewmouse"
pcutoff = 0.95
videotype = os.path.splitext(video_path_1)[1]
scale_list = []

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
dur_cutoff = 5
if duration > dur_cutoff:
    print(f"video duration too long, making {dur_cutoff} sec trimmed copy")
    # clean up the previous time it ran.
    # can be turned off, most useful for debugging
    for f in os.listdir("/tmp"):
        if "short" in f:
            os.remove(os.path.join("/tmp", f))
    vid_frames = 0
    fourcc = 0
    short_video_path_1 = "/tmp/short_1.avi"
    short_video_path_2 = "/tmp/short_2.avi"
    short_video_1 = cv2.VideoWriter(short_video_path_1, fourcc, fps, (w,h))
    short_video_2 = cv2.VideoWriter(short_video_path_2, fourcc, fps, (w,h))
    while(vid_frames < dur_cutoff * fps):
        ret, img_1 = video_1.read()
        ret, img_2 = video_2.read()
        if ret == False:
            assert False, "why are the video lengths different?"
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
    pseudo_threshold=0.9
)

path_camera_matrix = os.path.join(_tmp, save_dir, "camera_matrix/")
path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
stereo_info = stereo_file['camera-1-camera-2']
P1 = stereo_info['P1']
P2 = stereo_info['P2']

predictions_1 = os.path.join(video_name_1 + 'DLC_snapshot-1000.h5')
predictions_2 = os.path.join(video_name_2 + 'DLC_snapshot-1000.h5')

df_1 = pd.read_hdf(predictions_1)
df_2 = pd.read_hdf(predictions_2)

colormap = plt.get_cmap("jet")
col = colormap(np.linspace(0, 1, round(len(df_1.iloc[0]) / 3)))
markerSize = 18
markerType = '*'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
recon_path = '3d_reconstruction.avi'
vid = cv2.VideoWriter(recon_path, fourcc, fps, (w, h)) 

all_filtered_ids = []
for i in range(len(df_1)):
    x_1, x_2 = [], []
    y_1, y_2 = [], []
    prob_1, prob_2 = [], []
    frame = df_1.iloc[i]
    for j in range(0, len(frame), 3):
        x_1.append(frame[j])
        y_1.append(frame[j+1])
        prob_1.append(frame[j+2])

    frame = df_2.iloc[i]
    for j in range(0, len(frame), 3):
        x_2.append(frame[j])
        y_2.append(frame[j+1])
        prob_2.append(frame[j+2])

    # @TODO Probability weighted multiple camera voting?
    # @TODO convert to fake dlc3d .h5 output. Label missing pts 0 confidence

    filtered_ids = []
    for j, (p1, p2) in enumerate(zip(prob_1, prob_2)):
        if p1 < pcutoff or p2 < pcutoff:
            filtered_ids.append(j)
            all_filtered_ids.append(j)
    
    x1_and_y1 = cv2.undistortPoints(
        src=np.array(list(zip(x_1, y_1))),
        cameraMatrix=stereo_info["cameraMatrix1"],
        distCoeffs=stereo_info["distCoeffs1"],
        P=P1,
        R=stereo_info["R1"],
    )
    
    x_1 = np.squeeze(x1_and_y1[:,:,0])
    y_1 = np.squeeze(x1_and_y1[:,:,1])
    
    x2_and_y2 = cv2.undistortPoints(
        src=np.array(list(zip(x_2, y_2))),
        cameraMatrix=stereo_info["cameraMatrix2"],
        distCoeffs=stereo_info["distCoeffs2"],
        P=P2,
        R=stereo_info["R2"],
    )
    
    x_2 = np.squeeze(x2_and_y2[:,:,0])
    y_2 = np.squeeze(x2_and_y2[:,:,1])
    
    fig = plt.figure()
    num_possible = round(len(df_1.iloc[0]) / 3)
    num_detected = num_possible - len(filtered_ids)
    fig.title(f'{num_detected} out of {num_possible} possible pts detected above {pcutoff} confidence')
    ax = fig.add_subplot(111, projection="3d")
    ### @TODO automate this so it'll work on any dataset. Yoink from dlc
    ### don't worry about perfection, this is just verification for kypt-moseq
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 20)
    ax.set_zlim(26, 30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(-113, -200)
    for j in range(len(x_1)):
        if j in filtered_ids:
            continue
        cam1_pts = np.array([x_1[j], y_1[j]]).T
        cam2_pts = np.array([x_2[j], y_2[j]]).T
        _3d_pt = cv2.triangulatePoints(P1[:3], P2[:3], cam1_pts, cam2_pts)
        _3d_pt = (_3d_pt / _3d_pt[3])[:-1]
        xs = _3d_pt[0]
        ys = _3d_pt[1]
        zs = _3d_pt[2]
        ax.scatter(xs, ys, zs, c=col[j], marker=markerType, s=markerSize)
    plt.savefig("/tmp/_.png")
    plt.close()
    img = cv2.imread("/tmp/_.png")
    img = cv2.resize(img, (w, h))
    vid.write(img)
vid.release()

print("Keypoint filter counts:")
filtered_count = [0] * round(len(df_1.iloc[0]) / 3)
for i in all_filtered_ids:
    filtered_count[i] += 1
print(list(zip(range(len(filtered_count)), filtered_count)))

# combines the 3 videos side-by-side
combined_vid = cv2.VideoWriter('combined.avi', fourcc, fps, (w*3, h)) 
video_1 = cv2.VideoCapture(video_path_1)
video_2 = cv2.VideoCapture(video_path_2)
video_3 = cv2.VideoCapture(recon_path)
while(True):
    ret1, cam1 = video_1.read()
    ret2, cam2 = video_2.read()
    ret3, cam3 = video_3.read()
    if (not ret1) or (not ret2) or (not ret3):
        break
    cam3 = cv2.resize(cam3, (w,h))
    combined_img = np.concatenate((cam1, cam2, cam3), axis=1)
    combined_vid.write(combined_img)
video_1.release()
video_2.release()
video_3.release()
combined_vid.release()
