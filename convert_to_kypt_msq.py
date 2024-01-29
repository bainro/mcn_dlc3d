import os
import numpy as np
import keypoint_moseq as kpms

body_parts = [
    "cervical spine",
    "thoracic spine",
    "lumbar spine",
    "tail base",
    "head",
    "left ear",
    "right ear",
    "nose",
    "left hindpaw base",
    "left hindpaw tip",
    "right hindpaw base",
    "right hindpaw tip",
    "left forepaw tip",
    "right forepaw tip"
]

skeleton = [
    ['cervical spine', 'thoracic spine'], 
    ['thoracic spine', 'lumbar spine'], 
    ['lumbar spine', 'tail base'], 
    ['cervical spine', 'head'], 
    ['head', 'left ear'], 
    ['head', 'right ear'],
    ['head', 'nose'],
    ['left hindpaw base', 'left hindpaw tip'],
    ['left hindpaw base', 'thoracic spine'],
    ['right hindpaw base', 'right hindpaw tip'],
    ['right hindpaw base', 'thoracic spine'],
    ['left forepaw tip', 'cervical spine'],
    ['right forepaw tip', 'cervical spine'],
]
        
n_pts = len(body_parts)

_3d_kypts = pd.read_pickle(r'/home/rbain/git/mcn_dlc3d/kpms_3D_data.p')
### @TODO yoink syntax to read all vids in for loop
# single_m_vid = _3d_kypts['21_11_8_one_mouse']
asser False, "load the npy now instead"

confidences = np.ones((n_frames, n_pts)) 

project_dir = './tmp/kmoseq'
config = lambda: kpms.load_config(project_dir)
video_dir = os.path.join('/home/rbain/git/mcn_dlc3d/moseq_vids')
kpms.setup_project(
    project_dir,
    video_dir=video_dir,
    bodyparts=body_parts,
    skeleton=skeleton,
    overwrite=True
)

kpms.update_config(
    project_dir,
    anterior_bodyparts=['nose'],
    posterior_bodyparts=['tail base'],
    use_bodyparts=body_parts
)

### @TODO convert their pickle to an npy?
coordinates = np.load("coordinates.npy")
coordinates = {'only_recording': coordinates}

# format data for modeling
data, metadata = kpms.format_data(coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

kpms.update_config(project_dir, latent_dim=4)

# initialize the model
model = kpms.init_model(data, pca=pca, **config())

# optionally modify kappa
# model = kpms.update_hypparams(model, kappa=1e5)

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
