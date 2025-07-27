import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image, ImageSequence, ImageOps
from pathlib import Path
import numpy as np
import trimesh as Trimesh
from tqdm import tqdm
import gc

from .direct3d_s2.pipeline import Direct3DS2Pipeline

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.utils

script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_folders_os(path):
    """
    Returns a list of all immediate subdirectories (folders) within the given path.

    Args:
        path (str): The path to search for folders.

    Returns:
        list: A list of strings, where each string is the name of a folder.
              Returns an empty list if the path does not exist or contains no folders.
    """
    if not os.path.isdir(path):
        print(f"Error: Path '{path}' does not exist or is not a directory.")
        return []

    folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def parse_string_to_int_list(number_string):
  """
  Parses a string containing comma-separated numbers into a list of integers.

  Args:
    number_string: A string containing comma-separated numbers (e.g., "20000,10000,5000").

  Returns:
    A list of integers parsed from the input string.
    Returns an empty list if the input string is empty or None.
  """
  if not number_string:
    return []

  try:
    # Split the string by comma and convert each part to an integer
    int_list = [int(num.strip()) for num in number_string.split(',')]
    return int_list
  except ValueError as e:
    print(f"Error converting string to integer: {e}. Please ensure all values are valid numbers.")
    return []

def get_picture_files(folder_path):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    picture_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    picture_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
                
    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(picture_extensions):
                picture_files.append(full_path)                
    return picture_files
    
def get_mesh_files(folder_path, name_filter = None):
    """
    Retrieves all picture files (based on common extensions) from a given folder.

    Args:
        folder_path (str): The path to the folder to search.

    Returns:
        list: A list of full paths to the picture files found.
    """
    mesh_extensions = ('.obj', '.glb')
    mesh_files = []

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return []
                    
    for entry_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry_name)

        # Check if the entry is actually a file (and not a sub-directory)
        if os.path.isfile(full_path):
            file_name, file_extension = os.path.splitext(entry_name)
            if file_extension.lower().endswith(mesh_extensions):
                if name_filter is None or name_filter.lower() in file_name.lower():
                    mesh_files.append(full_path)                 
    return mesh_files    

def get_filename_without_extension_os_path(full_file_path):
    """
    Extracts the filename without its extension from a full file path using os.path.

    Args:
        full_file_path (str): The complete path to the file.

    Returns:
        str: The filename without its extension.
    """
    # 1. Get the base name (filename with extension)
    base_name = os.path.basename(full_file_path)
    
    # 2. Split the base name into root (filename without ext) and extension
    file_name_without_ext, _ = os.path.splitext(base_name)
    
    return file_name_without_ext

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
    
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def convert_tensor_images_to_pil(images):
    pil_array = []
    
    for image in images:
        pil_array.append(tensor2pil(image))
        
    return pil_array     

class Hy3DDirect3DS2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_path": (["wushuang98/Direct3D-S2"],{"default":"wushuang98/Direct3D-S2"}),
                "subfolder": (["direct3d-s2-v-1-0","direct3d-s2-v-1-1"],{"default":"direct3d-s2-v-1-1"}),
            },
        }

    RETURN_TYPES = ("HY3DS2PIPELINE", )
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline_path, subfolder):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        pipe = Direct3DS2Pipeline()
        pipe.init_config(pipeline_path, subfolder=subfolder)
        pipe.to(device)
        
        return (pipe,) 

class Hy3DRefineMeshWithDirect3DS2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DS2PIPELINE",),
                "image": ("IMAGE",),
                "trimesh": ("TRIMESH",),
                "sdf_resolution": ([512,1024],{"default":1024}),
                "steps": ("INT",{"default":15}),
                "guidance_scale": ("FLOAT",{"default":7.0,"min":0.0,"max":100.0}),
                "remove_interior": ("BOOLEAN",{"default":False}),
                "mc_threshold": ("FLOAT",{"default":0.2,"min":0.0,"max":1.0}),
                "seed": ("INT",{"default":0,"min":0,"max":0x7fffffff}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, image, trimesh, sdf_resolution, steps, guidance_scale, remove_interior, mc_threshold, seed):
        image = tensor2pil(image)
        
        if sdf_resolution==1024:
            trimesh = pipeline.refine_1024(image,trimesh,steps,guidance_scale,remove_interior,mc_threshold,seed)
        elif sdf_resolution==512:
            trimesh = pipeline.refine_512(image,trimesh,steps,guidance_scale,remove_interior,mc_threshold,seed)
        else:
            print(f'Unknown sdf_resolution: {sdf_resolution}')
        
        return (trimesh,)

class Hy3DRemoveInteriorWithDirect3DS2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DS2PIPELINE",),
                "trimesh": ("TRIMESH",),
                "sdf_resolution": ([512,1024],{"default":1024}),
                "mc_threshold": ("FLOAT",{"default":0.2,"min":0.0,"max":1.0}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, trimesh, sdf_resolution, mc_threshold):        
        if sdf_resolution==512:
            trimesh = pipeline.remove_interior_512(trimesh,mc_threshold)
        elif sdf_resolution==1024:
            trimesh = pipeline.remove_interior_1024(trimesh,mc_threshold)
        else:
            print(f'Unknown sdf_resolution: {sdf_resolution}')
        
        return (trimesh,)        
        

NODE_CLASS_MAPPINGS = {
    "Hy3DDirect3DS2ModelLoader": Hy3DDirect3DS2ModelLoader,
    "Hy3DRefineMeshWithDirect3DS2": Hy3DRefineMeshWithDirect3DS2,
    "Hy3DRemoveInteriorWithDirect3DS2": Hy3DRemoveInteriorWithDirect3DS2,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DDirect3DS2ModelLoader": "Hy3D Direct3DS2 Model Loader",
    "Hy3DRefineMeshWithDirect3DS2": "Hy3D Refine Mesh With Direct3DS2",
    "Hy3DRemoveInteriorWithDirect3DS2": "Hy3D Remove Interior With Direct3DS2",
    }
