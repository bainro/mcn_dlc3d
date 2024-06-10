import os
import cv2 
import time
import queue
import shutil
import datetime
import tempfile
import threading
import numpy as np
import multiprocessing as mp

# simple error callback for debugging processes
def ecb(e):
    assert False, print(e)

# allows us to update video's FPS in parallel
def fps_worker(wrong_fps_vid, true_fps):    
    # open video with wrong FPS
    vid = cv2.VideoCapture(wrong_fps_vid)
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_file = tempfile.NamedTemporaryFile(suffix=".avi") 
    fixed_vid = cv2.VideoWriter(temp_file.name, fourcc, true_fps, (w, h)) 
    while(True):
        ret, frame = vid.read()
        if ret:
            fixed_vid.write(frame)
        else:
            vid.release()
            fixed_vid.release()
            temp_file.close()
            os.remove(wrong_fps_vid)
            shutil.move(temp_file.name, wrong_fps_vid)
            break

# allows us to grab images from webcams in parallel
def cam_worker(cam_id, vid_name, fps, q):    
    # open the webcam file / stream
    if os.name == 'nt':
        cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(cam_id)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FPS, FPS)
    assert cam.isOpened(), "camera failed to open"
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid = cv2.VideoWriter(vid_name, fourcc, fps, (w, h)) 

    # first few frames are way earlier wrt correct, later frames
    for _ in range(10):
        ret, current_frame = cam.read()
    
    while True:
        msg = None
        try:
            msg = q.get_nowait()
        except:
            pass
        
        if msg == "stop":
            cam.release()
            vid.release()
            break
        elif msg == "capture":
            vid.write(current_frame)
 
        ret, current_frame = cam.read()
        assert ret, "camera thread worker crashed :("

if __name__ == "__main__":
    # extra safe cleanup 
    cv2.destroyAllWindows() 

    record_time = datetime.datetime.now()
    record_time = record_time.strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = "recorded_videos"
    save_dir = os.path.join(save_dir, record_time)
    os.makedirs(save_dir, exist_ok=True)

    FPS = 60 # naive assumption; corrected later

    # get all cameras 
    cam_sys_ids = []
    cams = []
    i = 0
    while(i < 10): 
        if os.name == 'nt':
            temp_camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            temp_camera = cv2.VideoCapture(i)
        is_cam = temp_camera.isOpened()
        if is_cam:
            cams.append(temp_camera)
            cam_sys_ids.append(i)
        i += 1

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
    
    for c in cams:
        c.release()
    del cams

    cam_ids = []
    num_cams = int(input("How many cameras do you want to setup? "))
    for cam in range(num_cams):   
        cam_id = int(input(f"which camera is #{cam+1}? "))
        cam_ids.append(cam_id-1)

    # list of camera workers and their queues
    qs = []
    cam_ws = []
    vid_files = []
    for c_i in cam_ids:
        q = queue.Queue()
        qs.append(q)    
        # create a camera worker
        sys_id = cam_sys_ids[c_i]
        vname = os.path.join(save_dir, f'camera_{c_i}.avi')
        vid_files.append(vname)
        w_args = (sys_id, vname, FPS, q)
        cam_th = threading.Thread(target=cam_worker, args=w_args, daemon=False)
        # start the parallel worker thread
        cam_th.start()
        cam_ws.append(cam_th)

    # let camera workers setup (e.g. throw away 1st few frames)
    time.sleep(2)
    
    frame_i = 0
    last_time = None
    start_time = time.time()

    cv2.imshow("PRESS 'q' TO STOP RECORDING", np.zeros((240,240,3)))

    todays_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(os.getcwd(), f'{todays_date}_camera.log')
    with open(log_file, 'w') as log:
        while(True): 
            # only add more capture commands if they've finished the old ones
            qs_empty = True
            for q in qs:
                qs_empty = qs_empty and q.empty()
            if not qs_empty:
                continue
            # tell all the cameras to record the latest image
            for q in qs:
                q.put("capture")
            frame_i += 1
            if not frame_i % FPS:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                log.write(f'{now}\n')
                print(f'{frame_i} frames recorded!')
                now = time.time()
                if last_time != None:
                    elapsed = now - last_time
                    print(f'last {FPS} frames took {elapsed:.1f} seconds\n')
                last_time = now
        
                potential_key = cv2.waitKey(1)
                if potential_key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    for q in qs:
                        q.put("stop")
                    # wait for all the cameras and videos to be released
                    for w in cam_ws:
                        w.join()                
                    break
        
    elapsed_t = time.time() - start_time
    print(f"True seconds recorded: {elapsed_t:.1f}")
    true_fps = round(frame_i / elapsed_t)
    print(f"True FPS: {true_fps:.1f}")

    print("Fixing recorded video's FPS...")
    start_time = time.time()

    try:
        mp.set_start_method('spawn')
    except:
        pass
    fps_pool = mp.Pool(num_cams)
    for vid_name in vid_files:
        args = [vid_name, true_fps]
        fps_pool.apply_async(func=fps_worker, args=args, error_callback=ecb)
    fps_pool.close()
    fps_pool.join()
    print(f"Time to fix video FPS: {time.time() - start_time:.1f} seconds")

    cv2.destroyAllWindows()
