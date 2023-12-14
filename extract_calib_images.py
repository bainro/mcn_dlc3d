import os
import sys
import cv2 
import time
import matplotlib.pyplot as plt

# extra safe cleanup 
cv2.destroyAllWindows() 

help_str = "pass both camera files on the command line (relative path)"
assert len(sys.argv) == 3, help_str
cam_file_1 = os.path.join(os.getcwd(), sys.argv[1])
cam_file_2 = os.path.join(os.getcwd(), sys.argv[2])

working_dir = os.path.join(os.getcwd(), f"tmp_{int(time.time())}")
os.mkdir(working_dir)
os.chdir(working_dir)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Half of the side length of the search window when refining detected checkerboard corners for subpixel accuracy.
search_window_size=(11, 11)
cbcol = 8 # number of checkerboard columns
cbrow = 6 # number of checkerboard rows

vids = []
for c_id, cam_file in enumerate([cam_file_1, cam_file_2]):
    vid = cv2.VideoCapture(cam_file)
    vids.append(vid)
    
print("extracting calibration images!")
good_img_c = 0
vid_frames = 0
    
while(True):
    ret_0, img_0 = vids[0].read()
    ret_1, img_1 = vids[1].read()
    assert ret_0 == ret_1, "unequal video lengths!"
    vid_frames += 1
    if not vid_frames % 20:
        print("FINISHED 20 MORE FRAMES")
    if ret_0:
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
            # Draw the corners and store the images
            _img_0 = cv2.drawChessboardCorners(img_0.copy(), (cbcol, cbrow), corners_0, ret_0)
            _img_1 = cv2.drawChessboardCorners(img_1.copy(), (cbcol, cbrow), corners_1, ret_1)
            f1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            f1.suptitle(
                str("Check that each checkboard image is identified consistently"),
                fontsize=25,
            )
            # Display images in RGB
            ax1.imshow(cv2.cvtColor(_img_0, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(_img_1, cv2.COLOR_BGR2RGB))
            f1.show()
            good_or_bad = input("good (g) or bad (b) checkerboard pair? ")
            if good_or_bad == 'g':
                fname_0 = f"camera-1-{good_img_c}.jpg"
                fname_1 = f"camera-2-{good_img_c}.jpg"
                cv2.imwrite(fname_0, img_0)
                cv2.imwrite(fname_1, img_1)
                good_img_c += 1
    else:
        vids[0].release()
        vids[1].release()
        break

cv2.destroyAllWindows() 