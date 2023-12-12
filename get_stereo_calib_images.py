import os
import cv2 
import time

# extra safe cleanup 
cv2.destroyAllWindows() 

# get all cameras 
cams = []
i = 0
while(i < 10): 
    if os.name == 'nt':
        temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    else:
        temp_camera = cv2.VideoCapture(i)
    i += 1
    # print("\n", temp_camera.isOpened(), "\n")
    is_cam = temp_camera.isOpened()
    if is_cam:
        cams.append(temp_camera)

print(f'Number of cameras detected: {len(cams)}')

font = cv2.FONT_HERSHEY_SIMPLEX
# show all cameras 
while(True): 
    for c in range(len(cams)):
        vid = cams[c]
        ret, frame = vid.read()
        cv2.putText(frame, f"{c+1}", (100, 100), font, 3, (255, 255, 255), 8)
        cv2.imshow(f"camera #{c+1}. (press q to quit)", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows() 

num_cams = int(input("How many cameras do you want to setup? "))
_cams = []
for cam in range(num_cams):   
    cam_id = int(input(f"which camera is #{cam+1}? "))
    _cams.append(cams[cam_id-1])
cams = _cams

print("\nNow save images for calibration.")
print("Aim for between 70 and 100 images.")
print("CONTROLS:\n q - quit\n s - save image\n r - record video\n")

record_after = False
save_img_i = 0
while(True): 
    imgs = []
    for c_i, c in enumerate(cams):
        ret, frame = c.read() 
        imgs.append(frame)
        frame_with_label = frame.copy()
        cv2.putText(frame_with_label, f"{c_i+1}", (100, 100), font, 3, (255, 255, 255), 8)
        cv2.imshow(f"camera #{c_i+1}. (s - save image) (r - record video)", frame_with_label) 
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
    FPS = 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for c_i, c in enumerate(cams):
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid = cv2.VideoWriter(f'camera_{c_i+1}.avi', fourcc, FPS, (w, h)) 
        vids.append(vid)
    
    
    frame_i = 0
    last_time = None
    start_time = time.time()
    while(True): 
        for i, (v, c) in enumerate(zip(vids, cams)): 
            ret, frame = c.read()
            v.write(frame)
        frame_i += 1
        if not frame_i % FPS:
            print(f'{frame_i} frames recorded!')
            now = time.time()
            if last_time != None:
                elapsed = now - last_time
                print(f'last {FPS} frames took {elapsed:.1f} seconds')
            last_time = now
        potential_key = cv2.waitKey(1)
        if potential_key & 0xFF == ord('q'):
            break
    elapsed_t = time.time() - start_time
    print(f"True seconds recorded: {elapsed_t:.1f}")
    true_fps = round(frame_i / elapsed_t)
    print(f"True FPS: {true_fps:.1f}")

    # make a copy of the videos with the correct FPS
    print("Fixing recorded video's FPS...")
    start_time = time.time()
    for c_i in range(len(cams)):
        # open video with wrong FPS
        vid = cv2.VideoCapture(f'camera_{c_i+1}.avi')
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # makes the video unique so it won't get overwritten
        fixed_vid = cv2.VideoWriter(f'camera_{c_i+1}_{int(time.time())}.avi', fourcc, true_fps, (w, h)) 
        while(True):
            ret, frame = vid.read()
            if ret:
                fixed_vid.write(frame)
            else:
                vid.release()
                fixed_vid.release()
                break
    print(f"Time to fix video FPS: {time.time() - start_time:.1f} seconds")

# cleanup
for c in cams:
    c.release()
for v in vids:
    v.release()

cv2.destroyAllWindows() 