from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import utils as utils
except:
    import scDMFK.utils as utils
    
import numpy as np
import h5py
import scipy as sp
import pandas as pd
import scanpy as sc # "scanpy.api" has been deprecated in newer versions


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utils.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utils.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=utils.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=utils.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


# def prepro(filename):
#     data_path = "data/" + filename + "/data.h5"
#     mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
#     if isinstance(mat, np.ndarray):
#         X = np.array(mat)
#     else:
#         X = np.array(mat.toarray())
#     cell_name = np.array(obs["cell_type1"])
#     cell_type, cell_label = np.unique(cell_name, return_inverse=True)
#     return X, cell_label
def prepro(filename, transpose=False):
    data_path = "data/" + filename

    try:
        adata = sc.read(data_path, first_column_names=True)
        adata.obs['Group'] = None
    except:
        mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
        if not isinstance(mat, np.ndarray):
            # X = np.array(mat.toarray())
            X = mat.toarray() #YD change
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)

        # X = np.ceil(X).astype(np.int)
        adata = sc.AnnData(X)
        adata.obs['Group'] = cell_label
    if transpose:
        adata = adata.transpose()
    print('Successfully preprocessed {} genes and {} cells'.format(adata.n_vars, adata.n_obs))
    return adata



def normalize(adata, highly_genes = None, highly_subset=False, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes, subset=highly_subset)

    return adata


def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                #   sep='\t',
                                                                index=(rownames is not None),
                                                                header=(colnames is not None),
                                                                float_format='%.6f')
