# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mrmeshpy
import trimesh
    
def postprocessmesh(vertices: np.array, faces: np.array, face_num: int = 200000, subdivided_parts: int = 16):
    num_faces = len(faces)
    
    print(f"Number of Vertices: {len(vertices)} - Number of Faces: {num_faces}")
    
    if face_num >= num_faces:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    else:                            
        print('Generating Meshlib Mesh ...')
        mesh = mrmeshnumpy.meshFromFacesVerts(faces, vertices)
        print('Loading Pack Optimally ...')
        mesh.packOptimally()
        
        settings = mrmeshpy.DecimateSettings()
        faces_to_delete = num_faces - face_num
        print(f'Number of Faces to delete: {faces_to_delete}')
        settings.maxDeletedFaces = faces_to_delete # Number of faces to be deleted
        settings.maxError = 0.05 # Maximum error when decimation stops
        settings.packMesh = True
        
        # Recommended to set to number of CPU cores or more available for the best performance
        settings.subdivideParts = subdivided_parts
        
        mrmeshpy.decimateMesh(mesh, settings)
        
        out_verts = mrmeshnumpy.getNumpyVerts(mesh)
        out_faces = mrmeshnumpy.getNumpyFaces(mesh.topology)
        
        mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces)   
        print(f"Reduced faces, resulting in {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")
        
    return mesh


