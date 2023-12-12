import os
import cv2 
import time

# extra safe cleanup 
cv2.destroyAllWindows() 

# get all cameras 
cams = []
i = 0
while(i < 10): 
    # print(f'i: {i}')
    if os.name == 'nt':
        temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    else:
        temp_camera = cv2.VideoCapture(i)
    i += 1
    print("\n", temp_camera.isOpened(), "\n")
    is_cam = temp_camera.isOpened()
    if is_cam:
        cams.append(temp_camera)

print(f'Number of cameras detected: {len(cams)}')

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
print("CONTROLS:\n q - quit\n s - save\n r - record videos\n")

record_after = False
save_img_i = 0
while(True): 
    imgs = []
    for c_i, c in enumerate(cams):
        ret, frame = c.read() 
        imgs.append(frame)
        cv2.imshow(f"camera #{c_i}. ('s' to save image) ('r' to record)", frame) 
    potential_key = cv2.waitKey(1)
    if potential_key & 0xFF == ord('q'): 
        break
    elif potential_key & 0xFF == ord('s'): 
        save_img_i += 1
        print(f"Number of calibration images saved: {save_img_i}")
        for c_id, img in enumerate(imgs):
            fname = f"camera-{c_id+1}-{save_img_i}.jpg"
            cv2.imwrite(fname, img)
    elif potential_key & 0xFF == ord('r'):
        record_after = True
        break        

vids = []
if record_after:
    FPS = 15
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for c_i, c in enumerate(cams):
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid = cv2.VideoWriter(f'camera_{c_i}.avi', fourcc, FPS, (w, h)) 
        vids.append(vid)
    
    
    next_frame = time.time() + 1 / FPS
    frame_i = 0
    while(True): 
        if time.time() > next_frame:
            for v, c in zip(vids, cams): 
            # for i, (v, c) in enumerate(zip(vids, cams)): 
                # start_t = time.time()
                ret, frame = c.read()
                v.write(frame)
                # print(f"camera #{i} took {(time.time() - start_t) * 1000} ms")
            frame_i += 1
            next_frame = time.time() + 1 / FPS
            if not frame_i % FPS:
                print(f'{int(frame_i // FPS)} seconds recorded!')
        potential_key = cv2.waitKey(1)
        if potential_key & 0xFF == ord('q'):
            break

# cleanup
for c in range(len(cams)):
    cams[c].release()
for v in vids:
    v.release()

cv2.destroyAllWindows() 