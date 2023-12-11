# import the opencv library 
import cv2 

# extra safe cleanup 
cv2.destroyAllWindows() 

# get all cameras 
cams = []
i = 0
while(True): 
    temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    i += 1
    no_cam = not temp_camera.isOpened()
    if no_cam:
        temp_camera.release();
        break
    else:
        cams.append(temp_camera)

# show all cameras 
while(True): 
    for c in range(len(cams)):
        vid = cams[c]
        ret, frame = vid.read() 
        cv2.imshow(f"camera #{c}. (press q to quit)", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# allow user to specify two!
cv2.destroyAllWindows() 
cam_1_id = int(input("which camera is #1? "))
cam_2_id = int(input("which camera is #2? "))
cam_1 = cams[cam_1_id]
cam_2 = cams[cam_2_id]
del cams
cams = [cam_1, cam_2]

print("\nNow save images for calibration.")
print("Aim for between 70 and 100 images.")
print("CONTROLS:\n q - quit\n s - save\n")

save_img_i = 0
while(True): 
    imgs = []
    for c in cams:
        ret, frame = c.read() 
        imgs.append(frame)
        cv2.imshow(f"camera #{c}. ('s' to save image) ('q' to quit)", frame) 
    potential_key = cv2.waitKey(1)
    if potential_key & 0xFF == ord('q'): 
        break
    elif potential_key & 0xFF == ord('s'): 
        save_img_i += 1
        print(f"Number of calibration images saved: {save_img_i}")
        for c_id, img in enumerate(imgs):
            fname = f"camera-{c_id+1}-{save_img_i}.jpg"
            cv2.imwrite(fname, img)
    
# cleanup
for c in range(len(cams)):
    cams[c].release()

cv2.destroyAllWindows() 