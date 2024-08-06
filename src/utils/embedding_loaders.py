
def h5py_loader(path, i, extra_features=None):
    if extra_features is None:
        return h5py.File(path, 'r')['feats'][:]
    else:
        return np.concatenate((h5py.File(path, 'r')['feats'][:], np.tile(extra_features[i],reps=np.array([h5py.File(path, 'r')['feats'].shape[0],1]))), axis=1)

def pt_loader(path, i, extra_features=None):
    x = torch.load(path)
    x = x.numpy()
    if extra_features is None:
        return x
    else:
        return np.concatenate((x, np.tile(extra_features[i],reps=[x.shape[0],1])), axis=1)


def zarr_loader(path, i, extra_features=None):
    root = zarr.open(path, "r")
    x = root['features'][:]
    if extra_features is None:
        return x
    else:
        return np.concatenate((x, np.tile(extra_features[i],reps=[x.shape[0],1])), axis=1)


def load_features(embedding, extra_features=None):
    if embedding == 'retccl':
        filetype = '.h5'
        path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(args.patch_size) + ''
        all_features = {file: h5py_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'ctranspath_tuned':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/moco_pretrained_features_all_tilesize_' + str(args.patch_size) + '_embeddings_768/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'UNI':
        filetype = '.zarr'
        path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/UNI_embeddings/'
        all_features = {file: zarr_loader(path_to_extracted_features + "/" + file + "_uni_embedding"+ filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.exists(path_to_extracted_features + "/" + file + "_uni_embedding"+ filetype)}
    elif embedding == 'UNI_256':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/UNI_' + str(args.patch_size) + '_1024_UKHD_FULL_dataset/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    else:
        raise ValueError("Unknown embedding type")
    return all_features
