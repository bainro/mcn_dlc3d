import os
import numpy as np
import keypoint_moseq as kpms

'''
##################
# skeleton parts #
##################
[0]  cervical spine
[1]  thoracic spine
[2]  lumbar spine
[3]  tail base
[4]  head
[5]  left ear
[6]  right ear
[7]  nose
[8]  left hindpaw base
[9]  left hindpaw tip
[10] right hindpaw base
[11] right hindpaw tip
[12] left forepaw tip
[13] right forepaw tip
'''

x, y, z = frame[:,0], frame[:,1], frame[:,2]
# c-spine & back one
x[:2], y[:2], z[:2]
# back one further
x[1:3], y[1:3], z[1:3]
# back one further
x[2:4], y[2:4], z[2:4]
# c-spine to head
[x[0], x[4]], [y[0], y[4]], [z[0], z[4]]
# head to left ear
x[4:6], y[4:6], z[4:6]
# head to right ear
[x[4], x[6]], [y[4], y[6]], [z[4], z[6]]
# head to nose
[x[4], x[7]], [y[4], y[7]], [z[4], z[7]]
# left hind limb to left foot
x[8:10], y[8:10], z[8:10]
# left hind limb to spine/trunk
[x[8], x[1]], [y[8], y[1]], [z[8], z[1]]
# right hind limg to right foot
x[10:12], y[10:12], z[10:12]
# right hind limb to spine/trunk
[x[10], x[1]], [y[10], y[1]], [z[10], z[1]]
# front left paw to spine/trunk
[x[12], x[0]], [y[12], y[0]], [z[12], z[0]]
# front right paw to spine/trunk
[x[13], x[0]], [y[13], y[0]], [z[13], z[0]]

### @TODO update skeleton! Still DLC superanimal...
skeleton = [
    ['c-spine', 't-spine'], 
    ['c-spine', 't-spine'], 
]

body_parts = []
# make set of those in skeleton
for bp1, bp2 in skeleton:
    body_parts.append(bp1)
    body_parts.append(bp2)
body_parts = list(set(body_parts))
        
n_pts = len(body_parts)

_3d_kypts = pd.read_pickle(r'/home/rbain/git/mcn_dlc3d/kpms_3D_data.p')

### @TODO yoink syntax to read all vids in for loop
# single_m_vid = _3d_kypts['21_11_8_one_mouse']

confidences = np.zeros((n_frames, n_pts)) 

for i in range(n_frames):
    frame = single_m_vid[i]
    x, y, z = frame[:,0], frame[:,1], frame[:,2]

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
