import os
import cv2
import time
import shutil
import numpy as np
from random import shuffle


def extract_calib(cam_file_1, cam_file_2, save_dir, max_imgs=200):
    print("\nExtracting calibration images!")
    # extra safe cleanup
    cv2.destroyAllWindows()

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Half of the side length of the search window when refining detected checkerboard corners for subpixel accuracy.
    search_window_size = (11, 11)
    cbcol = 8  # number of checkerboard columns
    cbrow = 6  # number of checkerboard rows

    vids = []
    for c_id, cam_file in enumerate([cam_file_1, cam_file_2]):
        vid = cv2.VideoCapture(cam_file)
        vids.append(vid)

    # determine the spatial order in which the checkerboard is typically detected
    num_to_detect = 20
    cam_1_1st_pt_xs, cam_2_1st_pt_xs = [], []
    cam_1_1st_pt_ys, cam_2_1st_pt_ys = [], []
    cam_1_nth_pt_xs, cam_2_nth_pt_xs = [], []
    cam_1_nth_pt_ys, cam_2_nth_pt_ys = [], []
    while(True):
        ret_0, img_0 = vids[0].read()
        ret_1, img_1 = vids[1].read()
        assert ret_0 and ret_1, "calibration videos do not have enough detections"
        gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
        ret_0, corners_0 = cv2.findChessboardCorners(
            gray_0, (cbcol, cbrow), None
        )
        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        ret_1, corners_1 = cv2.findChessboardCorners(
            gray_1, (cbcol, cbrow), None
        )
        if ret_0 and ret_1:
            corners_0 = cv2.cornerSubPix(
                gray_0, corners_0, search_window_size, (-1, -1), criteria
            )
            corners_1 = cv2.cornerSubPix(
                gray_1, corners_1, search_window_size, (-1, -1), criteria
            )

            cam_1_1st_pt_xs.append(corners_0[0][0][0]) 
            cam_2_1st_pt_xs.append(corners_1[0][0][0]) 
            cam_1_1st_pt_ys.append(corners_0[0][0][1]) 
            cam_2_1st_pt_ys.append(corners_1[0][0][1]) 
            cam_1_nth_pt_xs.append(corners_0[-1][0][0]) 
            cam_2_nth_pt_xs.append(corners_1[-1][0][0]) 
            cam_1_nth_pt_ys.append(corners_0[-1][0][1]) 
            cam_2_nth_pt_ys.append(corners_1[-1][0][1]) 

            if len(cam_1_1st_pt_xs) >= num_to_detect:
                vids[0].release()
                vids[1].release()
                break

    help_str = f'Error during initial calibration: {len(cam_1_1st_pt_xs)} != {num_to_detect}'
    assert len(cam_1_1st_pt_xs) == num_to_detect, help_str
    avg_cam_1_1st_x = np.mean(np.array(cam_1_1st_pt_xs))
    avg_cam_2_1st_x = np.mean(np.array(cam_2_1st_pt_xs))
    avg_cam_1_1st_y = np.mean(np.array(cam_1_1st_pt_ys))
    avg_cam_2_1st_y = np.mean(np.array(cam_2_1st_pt_ys))
    avg_cam_1_nth_x = np.mean(np.array(cam_1_nth_pt_xs))
    avg_cam_2_nth_x = np.mean(np.array(cam_2_nth_pt_xs))
    avg_cam_1_nth_y = np.mean(np.array(cam_1_nth_pt_ys))
    avg_cam_2_nth_y = np.mean(np.array(cam_2_nth_pt_ys))
    
    # how the first detected corner is typically oriented wrt the last corner
    cam_1_1st_left  = (avg_cam_1_1st_x < avg_cam_1_nth_x)
    cam_1_1st_above = (avg_cam_1_1st_y < avg_cam_1_nth_y)
    cam_2_1st_left  = (avg_cam_2_1st_x < avg_cam_2_nth_x)
    cam_2_1st_above = (avg_cam_2_1st_y < avg_cam_2_nth_y)
    # print(f'cam_1_1st_left: {cam_1_1st_left}')
    # print(f'cam_1_1st_above: {cam_1_1st_above}')
    # print(f'cam_2_1st_left: {cam_2_1st_left}')
    # print(f'cam_2_1st_above: {cam_2_1st_above}')

    good_img_c = 0
    vid_frames = 0
    
    tmp_dir = os.path.join(f"/tmp/extract_calib_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)
    old_calib_imgs = os.listdir(tmp_dir)
    for f in old_calib_imgs:
        os.remove(os.path.join(tmp_dir, f))
    
    vids = []
    for c_id, cam_file in enumerate([cam_file_1, cam_file_2]):
        vid = cv2.VideoCapture(cam_file)
        vids.append(vid)

    while(True):
        ret_0, img_0 = vids[0].read()
        ret_1, img_1 = vids[1].read()
        vid_frames += 1
        if not vid_frames % 200:
            print(f"Finished processing {vid_frames} frames")
        if not (ret_0 and ret_1):
            print(f'{good_img_c} calibration image pairs found!')
            vids[0].release()
            vids[1].release()
            break
        else:
            gray_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
            ret_0, corners_0 = cv2.findChessboardCorners(
                gray_0, (cbcol, cbrow), None
            )
            gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            ret_1, corners_1 = cv2.findChessboardCorners(
                gray_1, (cbcol, cbrow), None
            )
            if ret_0 and ret_1:
                corners_0 = cv2.cornerSubPix(
                    gray_0, corners_0, search_window_size, (-1, -1), criteria
                )
                corners_1 = cv2.cornerSubPix(
                    gray_1, corners_1, search_window_size, (-1, -1), criteria
                )
                
                # for the first camera
                cam_x_agreement = (cam_1_1st_left  == (corners_0[0][0][0] < corners_0[-1][0][0]))
                cam_y_agreement = (cam_1_1st_above == (corners_0[0][0][1] < corners_0[-1][0][1]))
                good_1st = cam_x_agreement and cam_y_agreement
                # for the second camera
                cam_x_agreement = (cam_2_1st_left  == (corners_1[0][0][0] < corners_1[-1][0][0]))
                cam_y_agreement = (cam_2_1st_above == (corners_1[0][0][1] < corners_1[-1][0][1]))
                good_2nd = cam_x_agreement and cam_y_agreement

                both_good = (good_1st and good_2nd)
                if both_good:
                    fname_0 = os.path.join(tmp_dir, f"camera-1-{good_img_c}.jpg")
                    fname_1 = os.path.join(tmp_dir, f"camera-2-{good_img_c}.jpg")
                    cv2.imwrite(fname_0, img_0)
                    cv2.imwrite(fname_1, img_1)
                    good_img_c += 1

    cv2.destroyAllWindows()
    os.rmdir(save_dir)
    shutil.copytree(tmp_dir, save_dir)
    if good_img_c > max_imgs:
        print(f'Keeping {max_imgs} of them.')
        print(f'\nThe original directory with all {good_img_c} images is here:\n{tmp_dir}\n')
        shuffled_indices = list(range(good_img_c))
        shuffle(shuffled_indices)
        rejects = shuffled_indices[max_imgs:]
        for r in rejects:
            fname = f"camera-1-{r}.jpg"
            os.remove(os.path.join(save_dir, fname))
            fname = f"camera-2-{r}.jpg"
            os.remove(os.path.join(save_dir, fname))