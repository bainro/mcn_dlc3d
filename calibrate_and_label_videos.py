'''
=== TODOs ===
- Remove the low confidence kypts. They used ~7 one time
- FUTURE: Anipose camera group calibration
- FUTURE: speed things up with parallel processing (eg cv2.findChBd)
- FUTURE: Fix & remove dlclibrary patch
- FUTURE: Make video_adapt more consistent
'''
import os
import cv2
import glob
import deeplabcut
import dlclibrary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from deeplabcut.utils import auxiliaryfunctions
from extract_calib_images import extract_calib
_help = "Wrong version of dlclibrary. Not 0.0.4"
assert dlclibrary.__version__ == '0.0.4', _help
_help = "Wrong version of deeplabcut. Not 2.3.5"
assert deeplabcut.__version__ == '2.3.5', _help

def rm_unpaired_calib(calib_dir):
    calib_imgs = os.listdir(calib_dir)
    n_unpaired_removed = 0
    for f in calib_imgs:
        if "camera-1" in f:
            buddy_f = f.replace("camera-1", "camera-2")
        else:
            buddy_f = f.replace("camera-2", "camera-1")
        buddy_f = os.path.join(calib_dir, buddy_f)
        if not os.path.exists(buddy_f):    
            os.remove(os.path.join(calib_dir, f))
            n_unpaired_removed += 1
    print(f'\n{n_unpaired_removed} unpaired calibration images removed!\n')
    
# There's a naming mismatch that I fixed for this 1 model & posted on my website
dlclib_path = dlclibrary.__path__
if type(dlclib_path) == type([]):
    dlclib_path = dlclib_path[0]
incorrect_yaml = os.path.join(dlclib_path, 'dlcmodelzoo/modelzoo_urls.yaml')
os.remove(incorrect_yaml)        
os.system(f"wget http://rkbain.com/modelzoo_urls.yaml -O '{incorrect_yaml}'")
    
video_dir = "recorded_videos"
vids = sorted(os.listdir(video_dir))
# filter out potential junk
keepers = []
for v in vids:
    if ("camera_" in v) and (".avi" in v):
        v = os.path.join(video_dir, v)
        keepers.append(v)
vids = keepers
assert len(vids) > 0, f"No videos found in '{os.getcwd()}/recorded_videos/'"
assert not len(vids) % 2, "For each camera there should be 1 calibration & 1 mouse video!"
    
base_project = 'stereo_test'
# gather all possible camera pairs based on the number of recorded videos found
extensions = []
for e in range(int(len(vids) / 2)):
    for _e in range(int(len(vids) / 2)):
        if e < _e:
            suffix = "_cam_" + str(e+1) + "+" + str(_e+1)
            extensions.append(suffix)

all_filtered_ids = []
for conf_i, ext in enumerate(extensions):
    project = base_project + ext
    labeler = 'rob_bain'
    
    tmp_dir = os.path.join("tmp/")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # this bit of code allows us to access the same project across multiple days
    old_project_dir = f'{project}-{labeler}-*-3d/'
    old_project_dir = glob.glob(os.path.join(tmp_dir, old_project_dir))
    assert len(old_project_dir) <= 1, "Error: More than 1 such project found!"
    if len(old_project_dir) == 0:
        os.chdir(tmp_dir)
        try:
            print("\nSome of the following output from deeplabcut is not applicable...\n")
            conf_path = deeplabcut.create_new_project_3d(project, labeler, num_cameras=2)
            project_path = '/'.join(conf_path.split('/')[:-1])
            old_project_dir.append(project_path)
            os.chdir("..")
        except:
            os.chdir("..")
    old_project_dir = old_project_dir[0]
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3] + str(day))
    date = dt.today().strftime("%Y-%m-%d")
    save_dir = f'{project}-{labeler}-{date}-3d/'
    new_project_dir = os.path.join(os.getcwd(), tmp_dir, save_dir)
    print(f'\nRenaming project directory {old_project_dir} to {new_project_dir}\n')
    os.rename(old_project_dir, new_project_dir)
    conf_path_3d = os.path.join(os.getcwd(), tmp_dir, save_dir, 'config.yaml')
    
    v1 = int(ext[-3])
    v2 = int(ext[-1])
    v1_i = (v1 - 1) * 2 + 1
    v2_i = (v2 - 1) * 2 + 1
    
    do_calib = input(f"\nDo you want to calibrate cameras {v1} & {v2}? [Y/n] ")
    do_calib = ('Y' in do_calib) or ('y' in do_calib)
    if do_calib:
        calib_1 = vids[v1_i]
        calib_2 = vids[v2_i]
        print(f'calibrating stereo parameters using {calib_1} & {calib_2}')
        calib_save_dir = os.path.join(os.getcwd(), tmp_dir, save_dir, 'calibration_images/')
        # clear out any old images in there
        old_calib_imgs = os.listdir(calib_save_dir)
        redo = True
        if len(old_calib_imgs) > 0:
            redo = input("\nOld calibration images found. Do you want to delete them? [Y/n] ")
            redo = ('Y' in redo) or ('y' in redo)
        if redo:
            for f in old_calib_imgs:
                os.remove(os.path.join(calib_save_dir, f))
            extract_calib(calib_1, calib_2, calib_save_dir, max_imgs=300)
        rm_unpaired_calib(calib_save_dir)
        # this is for when you look in the "corners" dir & remove some poorly detected images
        deeplabcut.calibrate_cameras(conf_path_3d, cbrow=6, cbcol=8, calibrate=True)
        corners = os.path.join(os.getcwd(), tmp_dir, save_dir, "corners/")
        print(f'\nOpen {corners} and remove images with poor or inconsistently detected checkerboards.')
        print("\nIf you delete camera-1-X_corner.jpg, this code will delete camera-2-X_corner.jpg for you.")
        corner_files = os.listdir(corners)
        num_imgs = len(corner_files)
        ready = input(f'\nAre you satisfied with the images in {corners} and ready to proceed? [Y/n] ')
        ready = ('Y' in ready) or ('y' in ready)
        assert ready, "Just re-run the script when you're ready :)"
        rm_unpaired_calib(corners)        
        # remove the corresponding calibration_images given the user removed corner images
        for f in os.listdir(calib_save_dir):
            corresponding_corner_f = f.replace('.jpg', '_corner.jpg')
            if not corresponding_corner_f in os.listdir(corners):
                os.remove(os.path.join(calib_save_dir, f))
        if num_imgs != len(os.listdir(corners)):
            print("\nCorner images were removed. Recalibrating!\n")
            deeplabcut.calibrate_cameras(conf_path_3d, cbrow=6, cbcol=8, calibrate=True)
        deeplabcut.check_undistortion(conf_path_3d, cbrow=6, cbcol=8)

    do_label = input(f"\nDo you want to label videos with mice in cameras {v1} & {v2}? [Y/n] ")
    do_label = ('Y' in do_label) or ('y' in do_label)
    if do_label:
        video_path_1 = vids[v1_i - 1]
        video_path_2 = vids[v2_i - 1]
        video_name_1 = os.path.splitext(video_path_1)[0]
        video_name_2 = os.path.splitext(video_path_2)[0]
        
        supermodel_name = "superanimal_topviewmouse"
        pcutoff = 0.2
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
        dur_cutoff = 480
        if duration > dur_cutoff:
            print(f"video duration too long, making {dur_cutoff} sec trimmed copy")
            '''
            # Remove last run's output.
            for f in os.listdir("/tmp"):
                if "short" in f:
                    os.remove(os.path.join("/tmp", f))
            '''
            vid_frames = 0
            fourcc = 0
            short_video_path_1 = f"/tmp/short_{v1}.avi"
            short_video_path_2 = f"/tmp/short_{v2}.avi"
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
        
        potential_preds_1 = os.path.join(video_name_1 + 'DLC_snapshot-1000.h5')
        potential_preds_2 = os.path.join(video_name_2 + 'DLC_snapshot-1000.h5')
        old_labels_found = os.path.exists(potential_preds_1) and os.path.exists(potential_preds_2)
        reuse_labels = False
        if old_labels_found:
            reuse_labels = input(f"\nOld labels for cameras {v1} & {v2} found. Reuse? [Y/n] ")
            reuse_labels = ('Y' in reuse_labels) or ('y' in reuse_labels)
        if not reuse_labels:
            deeplabcut.video_inference_superanimal(
                [video_path_1, video_path_2],
                supermodel_name,
                videotype=videotype,
                video_adapt=True,
                adapt_iterations=1000,
                scale_list=scale_list,
                pcutoff=pcutoff,
                pseudo_threshold=0.95
            )
        
        path_camera_matrix = os.path.join(tmp_dir, save_dir, "camera_matrix/")
        path_stereo_file = os.path.join(path_camera_matrix, "stereo_params.pickle")
        stereo_file = auxiliaryfunctions.read_pickle(path_stereo_file)
        stereo_info = stereo_file['camera-1-camera-2']
        P1 = stereo_info['P1']
        P2 = stereo_info['P2']
        
        # 200,000 for no fine-tuning? 1,000 for video_adapt-ed?
        # predictions_1 = os.path.join(video_name_1 + 'DLC_snapshot-200000.h5')
        # predictions_2 = os.path.join(video_name_2 + 'DLC_snapshot-200000.h5')
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
        
        n_frames = len(df_1)
        ### @TODO remove! Just for convert_to_kypt_msq.py testing!
        # n_pts = 27
        # coordinates = np.zeros((n_frames,n_pts,3))
        # confidences = np.zeros((n_frames,n_pts))            
        
        frames = []
        # @TODO can speed up this whole thing quite a bit probably...
        for i in range(n_frames):
            x_1, x_2 = [], []
            y_1, y_2 = [], []
            prob_1, prob_2 = [], []
            frame = []
            _frame = df_1.iloc[i]
            for j in range(0, len(_frame), 3):
                x_1.append(_frame.iloc[j])
                y_1.append(_frame.iloc[j+1])
                prob_1.append(_frame.iloc[j+2])
        
            _frame = df_2.iloc[i]
            for j in range(0, len(_frame), 3):
                x_2.append(_frame.iloc[j])
                y_2.append(_frame.iloc[j+1])
                prob_2.append(_frame.iloc[j+2])
        
            filtered_ids = []
            for j, (p1, p2) in enumerate(zip(prob_1, prob_2)):
                if (p1 < pcutoff) or (p2 < pcutoff):
                    ### @TODO remove! Just for convert_to_kypt_msq.py testing!
                    # prob_1[j], prob_2[j] = 0, 0
                    filtered_ids.append(j)
                    all_filtered_ids.append(j)
            
            x1_and_y1 = cv2.undistortPoints(
                src=np.array(list(zip(x_1, y_1))),
                cameraMatrix=stereo_info["cameraMatrix1"],
                distCoeffs=stereo_info["distCoeffs1"],
                P=P1,
                R=stereo_info["R1"],
            )    
            x2_and_y2 = cv2.undistortPoints(
                src=np.array(list(zip(x_2, y_2))),
                cameraMatrix=stereo_info["cameraMatrix2"],
                distCoeffs=stereo_info["distCoeffs2"],
                P=P2,
                R=stereo_info["R2"],
            )
            x_1 = np.squeeze(x1_and_y1[:,:,0])
            y_1 = np.squeeze(x1_and_y1[:,:,1])    
            x_2 = np.squeeze(x2_and_y2[:,:,0])
            y_2 = np.squeeze(x2_and_y2[:,:,1])
            
            frame = []
            for j in range(len(x_1)):
                ### @TODO uncomment! Just for convert_to_kypt_msq.py testing!
                if j in filtered_ids:
                    continue
                cam1_pts = np.array([x_1[j], y_1[j]]).T
                cam2_pts = np.array([x_2[j], y_2[j]]).T
                _3d_pt = cv2.triangulatePoints(P1[:3], P2[:3], cam1_pts, cam2_pts)
                _3d_pt = (_3d_pt / _3d_pt[3])[:-1]
                x = _3d_pt[0][0]
                y = _3d_pt[1][0]
                z = _3d_pt[2][0]
                ### @TODO remove! Just for convert_to_kypt_msq.py testing!
                # coordinates[i,j,:] = x,y,z
                # confidences[i,j] = (prob_1[j] + prob_2[j]) / 2
                frame.append([x,y,z,j])
            frames.append(frame)
    
        ### @TODO remove! Just for convert_to_kypt_msq.py testing!
        # np.save("coordinates.npy", coordinates)
        # np.save("confidences.npy", confidences)
        # assert False, "Debugging! Stop here."
    
        xs, ys, zs, cs = [], [], [], []
        for frame in frames:
            for pt in frame:
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(pt[2])
                cs.append(pt[3])
        # Trick to force equal aspect ratio of 3D plots
        minmax_x = np.nanpercentile(xs, q=[25, 75]).T
        minmax_y = np.nanpercentile(ys, q=[25, 75]).T
        minmax_z = np.nanpercentile(zs, q=[25, 75]).T
        minmax_x *= 1.1
        minmax_y *= 1.1
        minmax_z *= 1.1
        
        _rx = minmax_x[1] - minmax_x[0]
        _ry = minmax_y[1] - minmax_y[0]
        _rz = minmax_z[1] - minmax_z[0]
        
        minmax_range = np.array([_rx, _ry, _rz]).max() # / 2
        
        mid_x = np.mean(minmax_x)
        xlim = mid_x - minmax_range, mid_x + minmax_range
        mid_y = np.mean(minmax_y)
        ylim = mid_y - minmax_range, mid_y + minmax_range
        mid_z = np.mean(minmax_z)
        zlim = mid_z - minmax_range, mid_z + minmax_range
        
        for frame in frames:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            elev, azi = 258, 229
            view = (elev, azi)
            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.xaxis.grid(False)
            ax.view_init(view[0], view[1])
            ax.set_xlabel("X", fontsize=10)
            ax.set_ylabel("Y", fontsize=10)
            ax.set_zlabel("Z", fontsize=10)
            num_possible = 27 # hard-coded
            num_detected = len(frame)
            plt.title(f'{num_detected} out of {num_possible} possible pts detected above {pcutoff} confidence')
            for pt in frame:
                x, y, z, j = pt
                ax.scatter(x, y, z, c=col[j], marker=markerType, s=markerSize)
            fig.canvas.draw()
            img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            plt.close()
            img = cv2.resize(img, (w, h))
            vid.write(img)
        vid.release()
    
        # combines the 3 videos side-by-side
        combined_vid = cv2.VideoWriter(f'combined_{ext}.avi', fourcc, fps, (w*3, h)) 
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

print("Keypoint filter counts:")
filtered_count = [0] * round(len(df_1.iloc[0]) / 3)
for i in all_filtered_ids:
    filtered_count[i] += 1
ids = range(len(filtered_count))
ids_and_counts = zip(ids, filtered_count)
print(list(ids_and_counts))
filtered_count, ids = zip(*sorted(zip(filtered_count, ids)))
print("In order:")
print(list(zip(ids[::-1], filtered_count[::-1])))
    