import os
import cv2 
import pandas as pd


og_vids_dir = r"/home/rbain/git/mcn_dlc3d/moseq_vids/"
results_dir = r"/home/rbain/git/mcn_dlc3d/tmp/kmoseq/2024_01_30-13_58_52/results/"
src_vid_fs = [] # source video files
results = os.listdir(results_dir)
# keep these lists parallel
for i, r in enumerate(results):
    if not r.endswith(".csv"):
        continue
    results[i] = os.path.join(results_dir, r)
    # @TODO remove this hardcoded string (ie glob)
    r = r[:-3] + 'top.ir.mp4' 
    src_vid_fs.append(os.path.join(og_vids_dir, r))

output_dir = os.path.join(results_dir, "analysis_vids")
os.makedirs(output_dir, exist_ok=True)

# quick loop over all videos' csv rows to see how many syllables
# open up a video output for each syllable, append to list
num_syl = 0
csv_s = []
for vid_csv in results:
    if not vid_csv.endswith(".csv"):
        continue
    # conver to dataframe
    data = pd.read_csv(vid_csv)
    frames = data['syllable']
    csv_s.append(frames)
    max_syl_id = max(frames)
    if max_syl_id > num_syl:
        num_syl = max_syl_id
        
print(f'{num_syl} kypt-moseq syllable labels total.')

# make sure our assumption of same FPS holds
src_vids = []
copy_vids = []
# open video file to read from
vid = cv2.VideoCapture(src_vid_fs[0])
src_vids.append(vid)
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)
# create copy of src videos with syllable labels overlaid
copy_basepath = f'{os.path.basename(src_vid_fs[0])[:-4]}.mp4'
path = os.path.join(output_dir, copy_basepath)
v = cv2.VideoWriter(path, fourcc, fps, (w, h)) 
copy_vids.append(v)
for f in src_vid_fs[1:]:
    vid = cv2.VideoCapture(f)
    src_vids.append(vid)
    err_txt = 'We require all videos to have same FPS. '
    err_txt += f'{fps} != {vid.get(cv2.CAP_PROP_FPS)}'
    assert fps == vid.get(cv2.CAP_PROP_FPS), err_txt
    # create copy of src videos with syllable labels overlaid
    v_f = os.path.join(output_dir, f'{os.path.basename(f)[:-4]}.mp4')
    v = cv2.VideoWriter(v_f, fourcc, fps, (w, h)) 
    copy_vids.append(v)

syl_vids = []
for i in range(num_syl):
    # open video file to write this syllable to
    v_f = os.path.join(output_dir, f'syl_{i}.mp4')
    v = cv2.VideoWriter(v_f, fourcc, fps, (w, h)) 
    syl_vids.append(v)

# loop over each video
for i, src_v in enumerate(src_vids):
    syllables = csv_s[i]
    src_name = src_vid_fs[i]
    src_name = os.path.basename(src_name)
    src_name = src_name[:-11]
    for frame_i, s in enumerate(syllables):
        ret, frame = src_v.read()
        assert True, "video path error :("
        copy = frame.copy()
        cv2.putText(copy, f'Syllable #{s}', (10, 30), font, 1, (255, 255, 255), 2)
        copy_vids[i].write(copy)
        # turn frame number into video timestamp
        t = frame_i / fps
        mins = t // 60
        secs = t % 60
        ### @TODO add frames_buffer to give more temporal context?
          # if consecutive frames, then wait to do buffering (or similar logic)
        overlaid_txt = f'{src_name}: {mins:.1f}min {secs:.1f}secs'
        cv2.putText(frame, overlaid_txt, (10, 30), font, 1, (255, 255, 255), 2)
        syl_vids[s].write(frame)
    err_txt = 'each video frame needs a corresponding csv row'
    assert src_v.read()[0] == False, err_txt
    # cleanup now so we can check some results earlier
    copy_vids[i].release()
    src_vids[i].release()

# needs to wait to cleanup
for v in syl_vids:
    v.release()

# output videos are rather large, so we'll resize with bash's ffmpeg
### @TODO add ffmpeg to system requirements / dependencies
### @TODO use subprocess.run() instead of os.system()
output_vids = os.path.listdir(output_dir)
for i, f in output_vids:
    output_vids[i] = os.path.join(output_dir, f)
    
for f in output_vids:
    f_split = f.split(".")
    smaller_vid = f_split[:-1] + "final." + f_split[-1]
    os.system(f"ffmpeg -i {f} {smaller_vid}")
    os.remove(f)

    
# %% <-- separates sections in spyder IDE