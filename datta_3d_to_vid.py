import cv2
import numpy as np
import pandas as pd
# import keypoint_moseq as kpms
from matplotlib import pyplot as plt

_3d_kypts = pd.read_pickle(r'/home/rbain/git/mcn_dlc3d/kpms_3D_data.p')
single_m_vid = _3d_kypts['21_11_8_one_mouse']
# should be 14 3d pts
# test_frame = single_m_vid[-1,...]
# confidences = np.zeros((n_frames,n_pts))  

colormap = plt.get_cmap("jet")
num_pts = single_m_vid.shape[1]
col = colormap(np.linspace(0, 1, num_pts))

fps = 30
w, h = 480, 480
# to save space: ffmpeg -i visualize_datta_3d_kypts.avi video.mp4
vid = cv2.VideoWriter("/tmp/visualize_datta_3d_kypts.avi", 0, fps, (w,h))
num_frames = single_m_vid.shape[0]
# useful for debugging
# num_frames = min(5 * fps, num_frames)

'''
##################
# skeleton parts #
##################
[0]  cervical spine
[1]  thoracic spine
[2]  lumbar spine
[3]  tail base
[4]  head
[5]  left ear
[6]  right ear
[7]  nose
[8]  left hindpaw base
[9]  left hindpaw tip
[10] right hindpaw base
[11] right hindpaw tip
[12] left forepaw tip
[13] right forepaw tip
'''

for i in range(num_frames):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    frame = single_m_vid[i]
    x, y, z = frame[:,0], frame[:,1], frame[:,2]
    ax.scatter(x, y, z, c=col, s=100)
    ax.set_axis_off()
    # c-spine & back one
    ax.plot(x[:2], y[:2], z[:2], color='black')
    # back one further
    ax.plot(x[1:3], y[1:3], z[1:3], color='black')
    # back one further
    ax.plot(x[2:4], y[2:4], z[2:4], color='black')
    # c-spine to head
    ax.plot([x[0], x[4]], [y[0], y[4]], [z[0], z[4]], color='black')
    # head to left ear
    ax.plot(x[4:6], y[4:6], z[4:6], color='black')
    # head to right ear
    ax.plot([x[4], x[6]], [y[4], y[6]], [z[4], z[6]], color='black')
    # head to nose
    ax.plot([x[4], x[7]], [y[4], y[7]], [z[4], z[7]], color='black')
    # left hind limb to left foot
    ax.plot(x[8:10], y[8:10], z[8:10], color='black')
    # left hind limb to spine/trunk
    ax.plot([x[8], x[1]], [y[8], y[1]], [z[8], z[1]], color='black')
    # right hind limg to right foot
    ax.plot(x[10:12], y[10:12], z[10:12], color='black')
    # right hind limb to spine/trunk
    ax.plot([x[10], x[1]], [y[10], y[1]], [z[10], z[1]], color='black')
    # front left paw to spine/trunk
    ax.plot([x[12], x[0]], [y[12], y[0]], [z[12], z[0]], color='black')
    # front right paw to spine/trunk
    ax.plot([x[13], x[0]], [y[13], y[0]], [z[13], z[0]], color='black')
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    img = cv2.resize(img, (w, h))
    vid.write(img)
    plt.close()
vid.release()
