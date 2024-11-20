import zarr
import os
import h5py
import torch
import numpy as np
import pandas as pd

from src.utils.point_grid import PointGrid

def h5py_loader_with_coords(path, i, extra_features=None, feature_name='features'):
    """
    Loads features from an HDF5 file.

    Parameters:
    path (str): The path to the HDF5 file.
    i (int): The index of the extra features to include.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    numpy.ndarray: The loaded features, optionally concatenated with the extra features.
    """
    with h5py.File(path, 'r') as f:
        point_grid = PointGrid(f['pos_x'][:], f['pos_y'][:])
        if extra_features is None:
            features = f[feature_name][:]
        else:
            features = np.concatenate((f[feature_name][:], np.tile(extra_features[i],reps=np.array([f[feature_name].shape[0],1]))), axis=1)
        return features, point_grid


def h5py_loader(path, i, extra_features=None, feature_name='feats'):
    """
    Loads features from an HDF5 file.

    Parameters:
    path (str): The path to the HDF5 file.
    i (int): The index of the extra features to include.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    numpy.ndarray: The loaded features, optionally concatenated with the extra features.
    """
    if extra_features is None:
        return h5py.File(path, 'r')[feature_name][:]
    else:
        return np.concatenate((h5py.File(path, 'r')[feature_name][:], np.tile(extra_features[i],reps=np.array([h5py.File(path, 'r')[feature_name].shape[0],1]))), axis=1)

def pt_loader(path, i, extra_features=None):
    """
    Loads features from a PyTorch file.

    Parameters:
    path (str): The path to the PyTorch file.
    i (int): The index of the extra features to include.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    numpy.ndarray: The loaded features, optionally concatenated with the extra features.
    """
    x = torch.load(path)
    x = x.numpy()
    if extra_features is None:
        return x
    else:
        return np.concatenate((x, np.tile(extra_features[i],reps=[x.shape[0],1])), axis=1)


def zarr_loader(path, i, extra_features=None):
    """
    Loads features from a Zarr file.

    Parameters:
    path (str): The path to the Zarr file.
    i (int): The index of the extra features to include.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    numpy.ndarray: The loaded features, optionally concatenated with the extra features.
    """
    root = zarr.open(path, "r")
    x = root['features'][:]
    if extra_features is None:
        return x
    else:
        return np.concatenate((x, np.tile(extra_features[i],reps=[x.shape[0],1])), axis=1)


def load_features(embedding, patch_size, slide_annots, extra_features=None):
    """
    Loads features for a given embedding type.

    Parameters:
    embedding (str): The type of embedding to load features for. Can be 'retccl', 'ctranspath_tuned', 'UNI', or 'UNI_256'.
    patch_size (int): The size of the patches to load features for.
    slide_annots (object): An object containing slide annotations.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    dict: A dictionary where the keys are file names and the values are the loaded features.
    """
    if embedding == 'retccl':
        filetype = '.h5'
        path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(patch_size) + ''
        all_features = {file: h5py_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'ctranspath_tuned':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/moco_pretrained_features_all_tilesize_' + str(patch_size) + '_embeddings_768/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'UNI':
        filetype = '.zarr'
        path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/UNI_embeddings/'
        all_features = {file: zarr_loader(path_to_extracted_features + "/" + file + "_uni_embedding"+ filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.exists(path_to_extracted_features + "/" + file + "_uni_embedding"+ filetype)}
    elif embedding == 'UNI_256':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/UNI_' + str(patch_size) + '_1024_UKHD_FULL_dataset/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'ProvGigapath':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/ProvGigaPath_' + str(patch_size) + '_1536_UKHD_FULL_dataset/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'Virchow':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/Virchow_' + str(patch_size) + '_1280x2_UKHD/pt_files'
        all_features = {file: pt_loader(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    else:
        raise ValueError("Unknown embedding type")
    return all_features



def load_features_with_coords(embedding, patch_size, slide_annots, extra_features=None):
    """
    Loads features for a given embedding type.

    Parameters:
    embedding (str): The type of embedding to load features for. Can be 'retccl', 'ctranspath_tuned', 'UNI', or 'UNI_256'.
    patch_size (int): The size of the patches to load features for.
    slide_annots (object): An object containing slide annotations.
    extra_features (list, optional): A list of extra features to include in the loaded features. Defaults to None.

    Returns:
    dict: A dictionary where the keys are file names and the values are the loaded features.
    """
    filetype = '.h5'
    if embedding == 'UNI_256':
        path_to_extracted_features = '/home/p163v/histopathology/UKHD_Neuro/UNI_256'
        all_features = {file: h5py_loader_with_coords(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'ProvGigapath':
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/ProvGigaPath_' + str(patch_size) + '_1536_UKHD_FULL_dataset/h5_files'
        all_features = {file: h5py_loader_with_coords(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    elif embedding == 'Virchow':
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/Virchow_' + str(patch_size) + '_1280x2_UKHD/h5_files'
        all_features = {file: h5py_loader_with_coords(path_to_extracted_features + "/" + file + filetype, i, extra_features) for i, file in enumerate(slide_annots.file) if os.path.isfile(path_to_extracted_features + "/" + file + filetype)}
    else:
        raise ValueError("Unknown embedding type")
    return all_features



def load_single_features(embedding, patch_size, file, extra_features=None):
    if extra_features is not None:
        raise NotImplementedError
    if embedding == 'retccl':
        filetype = '.h5'
        path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(patch_size) + ''
        return h5py_loader(path_to_extracted_features + "/" + file + filetype, None, None)
    if embedding == 'ctranspath_tuned':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/moco_pretrained_features_all_tilesize_' + str(patch_size) + '_embeddings_768/pt_files'
        return pt_loader(path_to_extracted_features + "/" + file + filetype, None, None)
    if embedding == 'UNI':
        filetype = '.zarr'
        path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/UNI_embeddings/'
        return zarr_loader(path_to_extracted_features + "/" + file + "_uni_embedding"+ filetype, None, None)
    if embedding == 'UNI_256':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/UNI_' + str(patch_size) + '_1024_UKHD_FULL_dataset/pt_files'
        return pt_loader(path_to_extracted_features + "/" + file + filetype, None, None)
    if embedding == 'ProvGigapath':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/ProvGigaPath_' + str(patch_size) + '_1536_UKHD_FULL_dataset/pt_files'
        return pt_loader(path_to_extracted_features + "/" + file + filetype, None, None)
    if embedding == 'Virchow':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/Virchow_' + str(patch_size) + '_1280x2_UKHD/pt_files'
        return pt_loader(path_to_extracted_features + "/" + file + filetype, None, None)
    raise ValueError("Unknown embedding type")

def check_slide_exists(embedding, patch_size, slide):
    if embedding == 'retccl':
        filetype = '.h5'
        path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0540/p163v/UKHD_Neuro/RetCLL_Features/' + str(patch_size) + ''
        return os.path.isfile(path_to_extracted_features + "/" + slide + filetype)
    if embedding == 'ctranspath_tuned':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/moco_pretrained_features_all_tilesize_' + str(patch_size) + '_embeddings_768/pt_files'
        return os.path.isfile(path_to_extracted_features + "/" + slide + filetype)
    if embedding == 'UNI':
        filetype = '.zarr'
        path_to_extracted_features = '/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/UKHD_Neuro/UNI_embeddings/'
        return os.path.exists(path_to_extracted_features + "/" + slide + "_uni_embedding"+ filetype)
    if embedding == 'UNI_256':
        filetype = '.pt'
        path_to_extracted_features = '/dkfz/cluster/gpu/data/OE0585/p163v/UKHD_embeddings/UNI_' + str(patch_size) + '_1024_UKHD_FULL_dataset/pt_files'
        return os.path.isfile(path_to_extracted_features + "/" + slide + filetype)
    if embedding == 'ProvGigapath':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/ProvGigaPath_' + str(patch_size) + '_1536_UKHD_FULL_dataset/pt_files'
        return os.path.isfile(path_to_extracted_features + "/" + slide + filetype)
    if embedding == 'Virchow':
        filetype = '.pt'
        path_to_extracted_features = '/omics/odcf/analysis/OE0606_projects/pancancer_histopathology/analysis/shared_playground/CNS_classification/embeddings/Virchow_' + str(patch_size) + '_1280x2_UKHD/pt_files'
        return os.path.isfile(path_to_extracted_features + "/" + slide + filetype)
    else:
        raise ValueError("Unknown embedding type")
