import os
import cv2 
import time

# extra safe cleanup 
cv2.destroyAllWindows() 

os.makedirs("recorded_videos", exist_ok=True)
os.chdir("recorded_videos")

try:
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
    
    is_calib = input("\nIs this going to be a checkerboard calibration video? [Y/n]")
    is_calib = ('y' in is_calib) or ('Y' in is_calib)
    if is_calib:
        print("Don't rotate the board more than ~50 degrees in the video plane (ie left or right).")
        print("Otherwise the corner points might not be detected consistently across images.")
    
    vids = []
    FPS = 15 # naive assumption; corrected below
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for c_i, c in enumerate(cams):
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if is_calib:
            wrong_fps_vid = f'camera_{c_i+1}_calib.avi'
        else:
            wrong_fps_vid = f'camera_{c_i+1}.avi'
        vid = cv2.VideoWriter(wrong_fps_vid, fourcc, FPS, (w, h)) 
        vids.append(vid)

    frame_i = 0
    last_time = None
    start_time = time.time()

    # first few frames are way earlier wrt correct, later frames
    for _i in range(10):
        for i, (v, c) in enumerate(zip(vids, cams)): 
            ret, frame = c.read()
    
    while(True): 
        # @TODO can speed up a lot by making the screen grabs asynchronous instead 
        #       of waiting for each frame to finish before starting another
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
        # @TODO make interactive without slowing down FPS
        if frame_i > 300:#00:
            break
        # potential_key = cv2.waitKey(1)
        # if potential_key & 0xFF == ord('q'):
        #    break
    elapsed_t = time.time() - start_time
    print(f"True seconds recorded: {elapsed_t:.1f}")
    true_fps = round(frame_i / elapsed_t)
    print(f"True FPS: {true_fps:.1f}")
    
    # make a copy of the videos with the correct FPS
    print("Fixing recorded video's FPS...")
    start_time = time.time()
    for c_i in range(len(cams)):
        if is_calib:
            wrong_fps_vid = f'camera_{c_i+1}_calib.avi'
        else:
            wrong_fps_vid = f'camera_{c_i+1}.avi'
        # open video with wrong FPS
        vid = cv2.VideoCapture(wrong_fps_vid)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fixed_vid = cv2.VideoWriter(f'camera_{c_i+1}_{int(start_time)}.avi', fourcc, true_fps, (w, h)) 
        while(True):
            ret, frame = vid.read()
            if ret:
                fixed_vid.write(frame)
            else:
                vid.release()
                fixed_vid.release()
                os.remove(wrong_fps_vid)
                break
    print(f"Time to fix video FPS: {time.time() - start_time:.1f} seconds")
    
    # cleanup
    for c in cams:
        c.release()
    for v in vids:
        v.release()
    
    cv2.destroyAllWindows()
    os.chdir("..")
except:
    os.chdir("..")
