import os
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2 
import sys
import time
import queue
import shutil
import random
import datetime
import tempfile
import threading
import subprocess
import numpy as np
import multiprocessing as mp
# fallback to cmd line prompts when gui not available
try:
    gui = True
    import tkinter as tk
    from tkinter import filedialog
    gui_root = tk.Tk()
    # windows ('nt') vs linux
    if os.name == 'nt':
        gui_root.attributes('-topmost', True, '-alpha', 0)
    else:
        gui_root.withdraw()
except:
    gui = False

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
    temp_file = os.path.join(temp_file.name)
    fixed_vid = cv2.VideoWriter(temp_file, fourcc, true_fps, (w, h)) 
    while(True):
        ret, frame = vid.read()
        if ret:
            fixed_vid.write(frame)
        else:
            vid.release()
            fixed_vid.release()
            os.remove(wrong_fps_vid)
            shutil.move(temp_file, wrong_fps_vid)
            break

# allows us to grab images from webcams in parallel
def cam_worker(cam_id, vid_name, fps, q):    
    # open the webcam file / stream
    if os.name == 'nt':
        cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        # cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        cam = cv2.VideoCapture(cam_id)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # causes 30FPS bug on windows: https://tinyurl.com/5n8vncuy
        cam.set(cv2.CAP_PROP_FPS, fps)
    assert cam.isOpened(), "camera failed to open"
    
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 30)
    focus = 0.5
    # cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cam.set(cv2.CAP_PROP_SETTINGS, 1)
    # cam.set(cv2.CAP_PROP_FOCUS, focus)
    # from -2 (bright) to -11 (dark) for c920
    # cam.set(cv2.CAP_PROP_EXPOSURE, -2)
    
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
        # cam.set(cv2.CAP_PROP_FOCUS, focus)
        # print(cam.get(cv2.CAP_PROP_AUTOFOCUS))
        # print(cam.get(cv2.CAP_PROP_FOCUS))
        # print(cam.get(cv2.CAP_PROP_EXPOSURE))
        assert ret, "camera thread worker crashed :("        

def record(num_cams, cam_sys_ids, save_dir, FPS, main_q):
    # list of camera workers and their queues
    qs = []
    vid_files = []
    m = mp.Manager()
    cam_pool = mp.Pool(num_cams)
    for c_i in cam_ids:
        q = m.Queue()
        qs.append(q)    
        # create a camera worker
        sys_id = cam_sys_ids[c_i]
        vname = os.path.join(save_dir, f'camera_{c_i}.avi')
        vid_files.append(vname)
        w_args = (sys_id, vname, FPS, q)
        cam_pool.apply_async(func=cam_worker, args=w_args, error_callback=ecb)

    # let camera workers setup (e.g. throw away 1st few frames)
    print("Initializing...")
    time.sleep(2)
    print("Recording started!")
    print()
    print("You can open the video files and skip to the end to check on the progress.")
    print("The end of the video will not update automatically, but will require closing")
    print("and re-opening the video file. Note the FPS will likely be wrong. This is ")
    print("corrected post-recording.")
    
    frame_i = 0
    last_time = None
    start_time = time.time()

    print()
    print("Press ENTER to stop recording.")
    print()

    msg = None
    todays_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(save_dir, f'{todays_date}_every_{FPS}_frames.log')
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
            if frame_i % FPS == 0:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                log.write(f'{now}\n')
                # print(f'{frame_i} frames recorded!')
                now = time.time()
                if last_time != None:
                    elapsed = now - last_time
                    print(f'recording at {FPS / elapsed:.1f} FPS')
                last_time = now

            try:
                msg = main_q.get_nowait()
            except:
                pass
            if msg == "stop":
                for q in qs:
                    q.put("stop")
                # wait for all the cameras and videos to be released
                cam_pool.close()
                cam_pool.join()
                del cam_pool 
                break
    
    elapsed_t = time.time() - start_time
    print(f"True seconds recorded: {elapsed_t:.1f}")
    true_fps = round(frame_i / elapsed_t)
    print(f"True FPS: {true_fps:.1f}")

    print("Fixing recorded video's FPS...")
    start_time = time.time()

    fps_pool = mp.Pool(num_cams)
    for vid_name in vid_files:
        args = [vid_name, true_fps]
        fps_pool.apply_async(func=fps_worker, args=args, error_callback=ecb)
    fps_pool.close()
    fps_pool.join()
    print(f"Time to fix video FPS: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    # extra safe cleanup 
    cv2.destroyAllWindows() 

    try:
        mp.set_start_method('spawn')
    except:
        pass
    
    save_dir = "recorded_videos"
    if gui:
        print("\nA dialog box should appear. It might be in the background")
        save_dir = filedialog.askdirectory(title="Select directory to save outputs")
        # windows ('nt') vs linux
        if os.name == 'nt':
            gui_root.attributes('-topmost', True, '-alpha', 0)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.abspath(save_dir)

    record_time = datetime.datetime.now()
    record_time = record_time.strftime('%Y-%m-%d_%H-%M-%S')
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
    # assert len(cams) >= 5, 'not enough cameras detected!'

    # show each camera & ask which one it is
    # accept 1-5 and blank (extra camera) as answers
    cam_ids = [None, None, None, None, None]
    for c in range(len(cams)):
        vid = cams[c]
        ret, frame = vid.read()
        img_path = os.path.join(os.getcwd(), "cam_test.png")
        cv2.imwrite(img_path, frame)
        native_image_app = {'linux':'xdg-open',
                                  'win32':'explorer',
                                  'darwin':'open'}[sys.platform]
        subprocess.Popen([native_image_app, img_path])
        cam_num = input('Which camera # is this picture from? Leave blank if it is an extra, unused camera\n')
        valid_response = False
        valid_response = valid_response or cam_num == '1'
        valid_response = valid_response or cam_num == '2'
        valid_response = valid_response or cam_num == '3'
        valid_response = valid_response or cam_num == '4'
        valid_response = valid_response or cam_num == '5'
        valid_response = valid_response or cam_num == ""
        assert valid_response, f'{cam_num} is an invalid response'
        if cam_num != "":
            cam_ids[int(cam_num)-1] = c
    
    for i, v in enumerate(cam_ids[::-1]):
        if v == None:
            del cam_ids[4-i]

    for c in cams:
        c.release()
    del cams
    num_cams = len(cam_ids)

    # thread and queue allows us to get user input w/o hogging CPU
    q = queue.Queue()
    w_args = (num_cams, cam_sys_ids, save_dir, FPS, q)
    rec_th = threading.Thread(target=record, args=w_args, daemon=False)
    # start the parallel worker thread
    rec_th.start()
    input("") # wait for user to press ENTER
    print("\nShutting down recording... wait for post-processing to finish\n")
    q.put("stop")
    rec_th.join()
    print("\nDONE!")
    cv2.destroyAllWindows()
