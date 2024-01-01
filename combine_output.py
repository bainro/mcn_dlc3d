"""
Test script! Not optimized for anyone's convenience except my own.
@author: rbain
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deeplabcut.utils import auxiliaryfunctions

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

num_cams = 3
intrinsics = []
extrinsics = []
Rs = []
DCs = []
stereo_conf_paths = [
    "/home/rbain/git/mcn_dlc3d/tmp/stereo_test_cam_1+2-rob_bain-2023-12-28-3d/camera_matrix/stereo_params.pickle",
    "/home/rbain/git/mcn_dlc3d/tmp/stereo_test_cam_1+3-rob_bain-2023-12-28-3d/camera_matrix/stereo_params.pickle",
    "/home/rbain/git/mcn_dlc3d/tmp/stereo_test_cam_2+3-rob_bain-2023-12-28-3d/camera_matrix/stereo_params.pickle"
]
for c_i, p in enumerate(stereo_conf_paths):
    stereo_file = auxiliaryfunctions.read_pickle(p)
    stereo_info = stereo_file['camera-1-camera-2']
    P1 = stereo_info['P1']
    P2 = stereo_info['P2']
    R1 = stereo_info['R1']
    R2 = stereo_info['R2']
    DC1 = stereo_info["distCoeffs1"]
    DC2 = stereo_info["distCoeffs2"]
    # might be mapping these wrong...
    INT1 = stereo_info["cameraMatrix1"]
    INT2 = stereo_info["cameraMatrix1"]
    if   c_i == 0: # Add P1
        intrinsics.append(INT1)
        extrinsics.append(P1)
        Rs.append(R1)
        DCs.append(DC1)
    elif c_i == 1: # Add P3
        intrinsics.append(INT2)
        extrinsics.append(P2)
        Rs.append(R2)
        DCs.append(DC2)
    elif c_i == 2: # Add P2
        intrinsics = [intrinsics[0], INT1, intrinsics[1]]
        extrinsics = [extrinsics[0], P1, extrinsics[1]]
        Rs = [Rs[0], R1, Rs[1]]
        DCs = [DCs[0], DC1, DCs[1]]

preds_1 = pd.read_hdf("/tmp/short_1DLC_snapshot-1000.h5")
preds_2 = pd.read_hdf("/tmp/short_2DLC_snapshot-1000.h5")
preds_3 = pd.read_hdf("/tmp/short_3DLC_snapshot-1000.h5")

fps = 29
w, h = 480, 480
vid = cv2.VideoWriter("/tmp/combined_3d_kypts.avi", 0, fps, (w,h))
colormap = plt.get_cmap("jet")
# x, y, and z are the 3 DoF (i.e. variables, parameters)
col = colormap(np.linspace(0, 1, round(len(preds_1.iloc[0]) / 3)))
num_possible = round(len(preds_1.iloc[0]) / 3)

frame_cutoff = 290
# for each video frame
for i in range(len(preds_1)):
    if i > frame_cutoff:
        vid.release()
        assert False, "Program ended correctly :)"
        
    x_1, x_2, x_3 = [], [], []
    y_1, y_2, y_3 = [], [], []
    prob_1, prob_2, prob_3 = [], [], []
    
    frame = preds_1.iloc[i]
    for j in range(0, len(frame), 3):
        x_1.append(frame.iloc[j])
        y_1.append(frame.iloc[j+1])
        prob_1.append(frame.iloc[j+2])

    frame = preds_2.iloc[i]
    for j in range(0, len(frame), 3):
        x_2.append(frame.iloc[j])
        y_2.append(frame.iloc[j+1])
        prob_2.append(frame.iloc[j+2])
        
    frame = preds_3.iloc[i]
    for j in range(0, len(frame), 3):
        x_3.append(frame.iloc[j])
        y_3.append(frame.iloc[j+1])
        prob_3.append(frame.iloc[j+2])

    pcutoff = 0.6
    for j, (p1, p2, p3) in enumerate(zip(prob_1, prob_2, prob_3)):
        if (p1 < pcutoff):
            x_1[j] = np.nan
            y_1[j] = np.nan
        if (p2 < pcutoff):
            x_2[j] = np.nan
            y_2[j] = np.nan
        if (p3 < pcutoff):
            x_3[j] = np.nan
            y_3[j] = np.nan

    xy_1 = cv2.undistortPoints(
        src=np.array(list(zip(x_1, y_1))),
        cameraMatrix=intrinsics[0],
        distCoeffs=DCs[0],
        P=extrinsics[0],
        R=Rs[0]
    )
    
    xy_2 = cv2.undistortPoints(
        src=np.array(list(zip(x_2, y_2))),
        cameraMatrix=intrinsics[1],
        distCoeffs=DCs[1],
        P=extrinsics[1],
        R=Rs[1]
    )
    
    xy_3 = cv2.undistortPoints(
        src=np.array(list(zip(x_3, y_3))),
        cameraMatrix=intrinsics[2],
        distCoeffs=DCs[2],
        P=extrinsics[2],
        R=Rs[2]
    )
    
    x_1 = np.squeeze(xy_1[:,:,0])
    y_1 = np.squeeze(xy_1[:,:,1])
    x_2 = np.squeeze(xy_2[:,:,0])
    y_2 = np.squeeze(xy_2[:,:,1])
    x_3 = np.squeeze(xy_3[:,:,0])
    y_3 = np.squeeze(xy_3[:,:,1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # @TODO Use dlc code? Or center each frame & remove outliers?
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(26, 30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(-113, -200)
    num_filtered = 0
    for j in range(len(x_1)):
        n_points = 1
        out = np.empty((n_points, 3))
        out[:] = np.nan
        points = np.zeros((num_cams, n_points, 2))
        points[0, 0, 0], points[0, 0, 1] = x_1[j], y_1[j]
        points[1, 0, 0], points[1, 0, 1] = x_2[j], y_2[j]
        points[2, 0, 0], points[2, 0, 1] = x_3[j], y_3[j]
        cam_mats = np.array(extrinsics)
        for ip in range(n_points):
            # should be (num_cams, n_points, 2)
            xy_s = points[:, ip, :] 
            good = ~np.isnan(xy_s[:, 0])
            if np.sum(good) >= 2:
                x, y, z = triangulate_simple(xy_s[good], cam_mats[good])
                ax.scatter(x, y, z, c=col[j], marker='*', s=35)
            else:
                num_filtered += 1
    
    num_detected = num_possible - num_filtered
    plt.title(f'{num_detected} out of {num_possible} possible pts detected above {pcutoff} confidence')
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close()
    img = cv2.resize(img, (w, h))
    vid.write(img)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
'''
cam1_pts = np.array([x_1[j], y_1[j]]).T
cam2_pts = np.array([x_2[j], y_2[j]]).T
_3d_pt = cv2.triangulatePoints(P1[:3], P2[:3], cam1_pts, cam2_pts)
(x, y, z) = (_3d_pt / _3d_pt[3])[:-1]
print(f'x: {x}', f'y: {y}', f'z: {z}')
'''