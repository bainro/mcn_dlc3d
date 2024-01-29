"""
@author: rbain
"""
import os
import numpy as np
import keypoint_moseq as kpms
 
all_body_parts = [
    'nose', 'left_ear', 'right_ear', 'left_ear_tip',  # 3
    'right_ear_tip', 'left_eye', 'right_eye', 'neck', # 7
    'mid_back', 'mouse_center', 'mid_backend',        # 10
    'mid_backend2', 'mid_backend3', 'tail_base',      # 13
    'tail1', 'tail2', 'tail3', 'tail4', 'tail5',      # 18
    'left_shoulder', 'left_midside', 'left_hip',      # 21
    'right_shoulder', 'right_midside', 'right_hip',   # 24
    'tail_end', 'head_midpoint'                       # 26
]

# Tail removal was recommended by keypoint-moseq tutorial
low_conf_parts = [
    'tail1', 'tail2', 'tail3', 'tail4', 'tail5',
    'tail_end', 'left_ear_tip', 'left_hip', 
    'right_ear_tip', 'right_shoulder', 'left_shoulder',
    'left_midside'
] 

used_skeleton = []
full_skeleton=[
    ['nose', 'head_midpoint'], 
    ['left_eye', 'head_midpoint'],
    ['right_eye', 'head_midpoint'],
    ['left_ear_tip', 'left_ear'],
    ['left_ear', 'head_midpoint'],
    ['right_ear_tip', 'right_ear'],
    ['right_ear', 'head_midpoint'],
    ['head_midpoint', 'neck'],
    ['left_shoulder', 'neck'],
    ['right_shoulder', 'neck'],
    ['neck', 'mid_back'],
    ['mid_back', 'mouse_center'],
    ['left_midside', 'mouse_center'],
    ['right_midside', 'mouse_center'],
    ['mouse_center', 'mid_backend'],
    ['mid_backend', 'mid_backend2'],
    ['mid_backend2', 'mid_backend3'],
    ['left_hip', 'mid_backend3'],
    ['right_hip', 'mid_backend3'],
    ['mid_backend3', 'tail_base'],
    ['tail_base', 'tail1'],
    ['tail1', 'tail2'],
    ['tail2', 'tail3'],
    ['tail3', 'tail4'],
    ['tail4', 'tail5'],
    ['tail5', 'tail_end'],
]
for c in full_skeleton:
    has_low_conf_pt = (c[0] in low_conf_parts) or (c[1] in low_conf_parts)
    if not has_low_conf_pt:
        used_skeleton.append(c)

# remove low conf body part pts
used_body_parts = []
for bp in all_body_parts:
    if bp not in low_conf_parts:
        used_body_parts.append(bp)
        
n_pts = len(used_body_parts)
n_removed = len(all_body_parts) - len(used_body_parts)
print(f'{n_removed} pts removed. {n_pts} still included')
        
data_dir = "./"
project_dir = './tmp/kmoseq'
config = lambda: kpms.load_config(project_dir)
video_dir = os.path.join(data_dir, 'recorded_videos')

kpms.setup_project(
    project_dir,
    video_dir=video_dir,
    bodyparts=all_body_parts,
    skeleton=used_skeleton,
    overwrite=True
)

kpms.update_config(
    project_dir,
    anterior_bodyparts=['nose'],
    posterior_bodyparts=['tail_base'],
    use_bodyparts=used_body_parts
)

coordinates = np.load("coordinates.npy")
coordinates = {'only_recording': coordinates}
confidences = np.load("confidences.npy")
confidences = {'only_recording': confidences}

# format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

kpms.update_config(project_dir, latent_dim=10)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())

# optionally modify kappa
model = kpms.update_hypparams(model, kappa=1e5)

    num_ar_iters = 50

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir,
    ar_only=True, num_iters=num_ar_iters
)

# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters)

# modify kappa to maintain the desired syllable time-scale
model = kpms.update_hypparams(model, kappa=1e5)

# run fitting for an additional 500 iters
_model = kpms.fit_model(
    model, data, metadata, project_dir, model_name, ar_only=False,
    start_iter=current_iter, num_iters=current_iter+500
)
model = _model[0]

# modify a saved checkpoint so syllables are ordered by frequency
kpms.reindex_syllables_in_checkpoint(project_dir, model_name);

# load the most recent model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results
results = kpms.extract_results(model, metadata, project_dir, model_name)

kpms.save_results_as_csv(results, project_dir, model_name)

results = kpms.load_results(project_dir, model_name)
kpms.generate_trajectory_plots(coordinates, results, project_dir, model_name, **config())
# Code below only works for 2D data. For 3D data:
# https://keypoint-moseq.readthedocs.io/en/latest/FAQs.html#making-grid-movies-for-3d-data
# kpms.generate_grid_movies(results, project_dir, model_name, coordinates=coordinates, **config());
# kpms.plot_similarity_dendrogram(coordinates, results, project_dir, model_name, **config())
