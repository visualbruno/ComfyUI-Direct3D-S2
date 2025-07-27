import torch
import numpy as np  
import udf_ext


def compute_valid_udf(vertices, faces, dim=512, threshold=8.0):
    if not faces.is_cuda or not vertices.is_cuda:
        raise ValueError("Both maze and visited tensors must be CUDA tensors")
    udf = torch.zeros(dim**3,device=vertices.device).int() + 10000000
    n_faces = faces.shape[0]
    udf_ext.compute_valid_udf(vertices, faces, udf, n_faces, dim, threshold)
    return udf.float()/10000000.

def normalize_mesh(mesh, scale=0.95):
    vertices = mesh.vertices
    min_coords, max_coords = vertices.min(axis=0), vertices.max(axis=0)
    dxyz = max_coords - min_coords
    dist = max(dxyz)
    mesh_scale = 2.0 * scale / dist
    mesh_offset = -(min_coords + max_coords) / 2
    vertices = (vertices + mesh_offset) * mesh_scale
    mesh.vertices = vertices
    return mesh

def mesh2index(mesh, size=1024, factor=8):
    vertices = torch.Tensor(mesh.vertices).float().cuda() * 0.5
    faces = torch.Tensor(mesh.faces).int().cuda()
    sdf = compute_valid_udf(vertices, faces, dim=size, threshold=4.0)
    sdf = sdf.reshape(size, size, size).unsqueeze(0)

    sparse_index = (sdf < 4/size).nonzero()
    sparse_index[..., 1:] = sparse_index[..., 1:] // factor
    latent_index = torch.unique(sparse_index, dim=0)
    return latent_index