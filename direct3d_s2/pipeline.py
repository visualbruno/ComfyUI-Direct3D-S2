import os
import torch
import numpy as np
import trimesh as Trimesh
import gc
import sys
from typing import Any
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from typing import Union, List, Optional

comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
direct3ds2_path = os.path.join(comfy_path, "models", "wushuang98", "Direct3D-S2")
config_path = os.path.join(comfy_path, "custom_nodes", "ComfyUI-Direct3D-S2", "config", "config.yaml")
current_folder = os.path.join(comfy_path, "custom_nodes", "ComfyUI-Direct3D-S2")

print(f'Adding to PATH : {current_folder}')
sys.path.append(current_folder)

from direct3d_s2.modules import sparse as sp
from direct3d_s2.utils import (
    instantiate_from_config, 
    preprocess_image, 
    sort_block, 
    extract_tokens_and_coords,
    normalize_mesh,
    mesh2index,
)

class Direct3DS2Pipeline(object):

    def __init__(self, device):
        self.dtype=torch.float16
        self.device = torch.device(device)  
        print(f'Comfy_path: {comfy_path}')

    def init_config(self, pipeline_path, subfolder):        
        dtype=torch.float16
        model_dir = os.path.join(direct3ds2_path, subfolder) 
        #self.config_path = config_path
        if os.path.isdir(model_dir):
            print(f'Model dir found. Using {subfolder}')
            self.config_path = os.path.join(model_dir, 'config.yaml')
            self.model_dense_path = os.path.join(model_dir, 'model_dense.ckpt')
            self.model_sparse_512_path = os.path.join(model_dir, 'model_sparse_512.ckpt')
            self.model_sparse_1024_path = os.path.join(model_dir, 'model_sparse_1024.ckpt')
            self.model_refiner_path = os.path.join(model_dir, 'model_refiner.ckpt')
            self.model_refiner_1024_path = os.path.join(model_dir, 'model_refiner_1024.ckpt')
        else:
            print('Model dir not found. Downloading from huggingface ...')
                        
            self.config_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder, 
                filename="config.yaml", 
                repo_type="model"
            )                        
                        
            self.model_dense_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_dense.ckpt", 
                repo_type="model"
            )
            self.model_sparse_512_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_sparse_512.ckpt", 
                repo_type="model"
            )
            self.model_sparse_1024_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_sparse_1024.ckpt", 
                repo_type="model"
            )
            self.model_refiner_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_refiner.ckpt", 
                repo_type="model"
            )
            self.model_refiner_1024_path = hf_hub_download(
                repo_id=pipeline_path, 
                subfolder=subfolder,
                filename="model_refiner_1024.ckpt", 
                repo_type="model"
            )

        self.cfg = OmegaConf.load(self.config_path)        

        self.dense_vae = None
        self.dense_dit = None
        self.sparse_vae_512 = None
        self.sparse_dit_512 = None
        self.sparse_vae_1024 = None
        self.sparse_dit_1024 = None
        self.refiner = None
        self.refiner_1024 = None
        self.dense_image_encoder = None
        self.sparse_image_encoder = None
        self.dense_scheduler = None
        self.sparse_scheduler_512 = None
        self.sparse_scheduler_1024 = None
        self.dtype = dtype
        
    def clear_memory(self):
        if self.dense_vae != None:
            del self.dense_vae
            self.dense_vae = None
            
        if self.dense_dit != None:
            del self.dense_dit
            self.dense_dit = None
            
        if self.sparse_vae_512 != None:
            del self.sparse_vae_512
            self.sparse_vae_512 = None
            
        if self.sparse_dit_512 != None:
            del self.sparse_dit_512
            self.sparse_dit_512 = None
            
        if self.sparse_vae_1024 != None:
            del self.sparse_vae_1024
            self.sparse_vae_1024 = None
            
        if self.sparse_dit_1024 != None:
            del self.sparse_dit_1024
            self.sparse_dit_1024 = None
            
        if self.refiner != None:
            del self.refiner
            self.refiner = None
            
        if self.refiner_1024 != None:
            del self.refiner_1024
            self.refiner_1024 = None
            
        if self.dense_image_encoder != None:
            del self.dense_image_encoder
            self.dense_image_encoder = None
            
        if self.sparse_image_encoder != None:
            del self.sparse_image_encoder
            self.sparse_image_encoder = None
            
        torch.cuda.empty_cache()
        gc.collect()

    def preprocess(self, image):
        if image.mode == 'RGBA':
            image = np.array(image)
        else:
            if getattr(self, 'birefnet_model', None) is None:
                from .utils import BiRefNet
                self.birefnet_model = BiRefNet(self.device)
            image = self.birefnet_model.run(image)
        image = preprocess_image(image)
        return image

    def prepare_image(self, image: Union[str, List[str], Image.Image, List[Image.Image]]):
        if not isinstance(image, list):
            image = [image]
        if isinstance(image[0], str):
            image = [Image.open(img) for img in image]
        image = [self.preprocess(img) for img in image]
        image = torch.stack([img for img in image]).to(self.device)
        return image
    
    def encode_image(self, image: torch.Tensor, conditioner: Any, 
                     do_classifier_free_guidance: bool = True, use_mask: bool = False):
        if use_mask:
            cond = conditioner(image[:, :3], image[:, 3:])
        else:
            cond = conditioner(image[:, :3])

        if isinstance(cond, tuple):
            cond, cond_mask = cond
            cond, cond_coords = extract_tokens_and_coords(cond, cond_mask)
        else:
            cond_mask, cond_coords = None, None

        if do_classifier_free_guidance:
            uncond = torch.zeros_like(cond)
        else:
            uncond = None
        
        if cond_coords is not None:
            cond = sp.SparseTensor(cond, cond_coords.int())
            if uncond is not None:
                uncond = sp.SparseTensor(uncond, cond_coords.int())

        return cond, uncond

    def inference(
            self,
            image,
            vae,
            dit,
            conditioner,
            scheduler,
            num_inference_steps: int = 30, 
            guidance_scale: int = 7.0, 
            generator: Optional[torch.Generator] = None,
            latent_index: torch.Tensor = None,
            mode: str = 'dense', # 'dense', 'sparse512' or 'sparse1024
            remove_interior: bool = False,
            mc_threshold: float = 0.02):
        
        do_classifier_free_guidance = guidance_scale > 0
        if mode == 'dense':
            sparse_conditions = False
        else:
            sparse_conditions = dit.sparse_conditions
        cond, uncond = self.encode_image(image, conditioner, 
                                         do_classifier_free_guidance, sparse_conditions)
        batch_size = cond.shape[0]

        if mode == 'dense':
            latent_shape = (batch_size, *dit.latent_shape)
        else:
            latent_shape = (len(latent_index), dit.out_channels)
        
        latents = torch.randn(latent_shape, dtype=self.dtype, device=self.device, generator=generator)            

        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps

        extra_step_kwargs = {
            "generator": generator
        }

        for i, t in enumerate(tqdm(timesteps, desc=f"{mode} Sampling:")):
            latent_model_input = latents
            timestep_tensor = torch.tensor([t], dtype=latent_model_input.dtype, device=self.device)

            if mode == 'dense':
                x_input = latent_model_input
            elif mode in ['sparse512', 'sparse1024']:
                x_input = sp.SparseTensor(latent_model_input, latent_index.int())

            diffusion_inputs = {
                "x": x_input,
                "t": timestep_tensor,
                "cond": cond,
            }

            noise_pred_cond = dit(**diffusion_inputs)
            if mode != 'dense':
                noise_pred_cond = noise_pred_cond.feats

            if do_classifier_free_guidance:
                diffusion_inputs["cond"] = uncond
                noise_pred_uncond = dit(**diffusion_inputs)
                if mode != 'dense':
                    noise_pred_uncond = noise_pred_uncond.feats
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
            
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        
        latents = 1. / vae.latents_scale * latents + vae.latents_shift
        
        if mode != 'dense':
            latents = sp.SparseTensor(latents, latent_index.int())
        
        decoder_inputs = {
            "latents": latents,
            "mc_threshold": mc_threshold,
        }
        if mode == 'dense':
            decoder_inputs['return_index'] = True
        elif remove_interior:
            decoder_inputs['return_feat'] = True
        if mode == 'sparse1024':
            decoder_inputs['voxel_resolution'] = 1024      
        
        outputs = vae.decode_mesh(**decoder_inputs)
        
        if remove_interior:            
            del latents, noise_pred, noise_pred_cond, noise_pred_uncond, x_input, cond, uncond
            self.clear_memory()
            
            if mode == 'sparse512':
                self.init_refiner()
                outputs = self.refiner.run(*outputs, mc_threshold=mc_threshold*2.0)
            elif mode == 'sparse1024':
                self.init_refiner_1024()                
                outputs = self.refiner_1024.run(*outputs, mc_threshold=mc_threshold)

        return outputs
        
    def init_refiner(self):
        state_dict_refiner = torch.load(self.model_refiner_path, map_location='cpu', weights_only=True)
        self.refiner = instantiate_from_config(self.cfg.refiner)
        self.refiner.load_state_dict(state_dict_refiner["refiner"], strict=True)
        self.refiner.eval()
        
        self.refiner.to(self.device)
        
    def init_refiner_1024(self):
        state_dict_refiner_1024 = torch.load(self.model_refiner_1024_path, map_location='cpu', weights_only=True)
        self.refiner_1024 = instantiate_from_config(self.cfg.refiner_1024)
        self.refiner_1024.load_state_dict(state_dict_refiner_1024["refiner"], strict=True)
        self.refiner_1024.eval()
        
        self.refiner_1024.to(self.device)

    def init_sparse_512(self):
        state_dict_sparse_512 = torch.load(self.model_sparse_512_path, map_location='cpu', weights_only=True)
        self.sparse_vae_512 = instantiate_from_config(self.cfg.sparse_vae_512) 
        self.sparse_vae_512.load_state_dict(state_dict_sparse_512["vae"], strict=True)
        self.sparse_vae_512.eval()
        self.sparse_vae_512.to(self.device)
        self.sparse_dit_512 = instantiate_from_config(self.cfg.sparse_dit_512)                                     
        self.sparse_dit_512.load_state_dict(state_dict_sparse_512["dit"], strict=True)
        self.sparse_dit_512.eval()
        self.sparse_dit_512.to(self.device)
        
        self.sparse_scheduler_512 = instantiate_from_config(self.cfg.sparse_scheduler_512)
        
        self.sparse_image_encoder = instantiate_from_config(self.cfg.sparse_image_encoder)
        self.sparse_image_encoder.to(self.device)
        
    def init_sparse_1024(self):
        state_dict_sparse_1024 = torch.load(self.model_sparse_1024_path, map_location='cpu', weights_only=True)
        self.sparse_vae_1024 = instantiate_from_config(self.cfg.sparse_vae_1024)                                        
        self.sparse_vae_1024.load_state_dict(state_dict_sparse_1024["vae"], strict=True)
        self.sparse_vae_1024.eval()
        self.sparse_vae_1024.to(self.device)
        self.sparse_dit_1024 = instantiate_from_config(self.cfg.sparse_dit_1024)
        self.sparse_dit_1024.load_state_dict(state_dict_sparse_1024["dit"], strict=True)
        self.sparse_dit_1024.eval()
        self.sparse_dit_1024.to(self.device)

        self.sparse_scheduler_1024 = instantiate_from_config(self.cfg.sparse_scheduler_1024)
        
        self.sparse_image_encoder = instantiate_from_config(self.cfg.sparse_image_encoder)
        self.sparse_image_encoder.to(self.device)
        
    def init_dense(self):
        state_dict_dense = torch.load(self.model_dense_path, map_location='cpu', weights_only=True)
        self.dense_vae = instantiate_from_config(self.cfg.dense_vae)
        self.dense_vae.load_state_dict(state_dict_dense["vae"], strict=True)
        self.dense_vae.eval()
        self.dense_vae.to(self.device)
        self.dense_dit = instantiate_from_config(self.cfg.dense_dit)
        self.dense_dit.load_state_dict(state_dict_dense["dit"], strict=True)
        self.dense_dit.eval()
        self.dense_dit.to(self.device)
        
        self.dense_scheduler = instantiate_from_config(self.cfg.dense_scheduler)        
        
        self.sparse_image_encoder = instantiate_from_config(self.cfg.sparse_image_encoder)
        self.sparse_image_encoder.to(self.device)
        
        
    @torch.no_grad()
    def refine_1024(self, image, mesh, steps, guidance_scale, remove_interior, mc_threshold, seed):
        self.clear_memory()
        self.init_sparse_1024()

        image = self.prepare_image(image)
            
        generator=torch.Generator(device=self.device).manual_seed(seed)
                
        scale = 0.97
        
        # while True:
            # mesh = normalize_mesh(mesh, scale=scale)
            # latent_index = mesh2index(mesh, size=1024, factor=8)
            # latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size) 
            # print(f"number of latent tokens: {len(latent_index)}")

            # if len(latent_index) <= 120_000:
                # break
            
            # scale -= 0.01 

        mesh = normalize_mesh(mesh, scale=scale)
        latent_index = mesh2index(mesh, size=1024, factor=8)
        latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size) 
        print(f"number of latent tokens: {len(latent_index)}")            
        
        mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024, 
                            self.sparse_image_encoder, self.sparse_scheduler_1024, 
                            generator=generator, mode='sparse1024', 
                            mc_threshold=mc_threshold, latent_index=latent_index, 
                            remove_interior=remove_interior, num_inference_steps=steps, guidance_scale=guidance_scale)[0]         
        return mesh
        
    @torch.no_grad()
    def refine_512(self, image, mesh, steps, guidance_scale, remove_interior, mc_threshold, seed):
        self.clear_memory()
        self.init_sparse_512()    
            
        generator=torch.Generator(device=self.device).manual_seed(seed)
                
        mesh = normalize_mesh(mesh)
        latent_index = mesh2index(mesh, size=512, factor=8)
        latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size) 

        image = self.prepare_image(image)

        mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512, 
                            self.sparse_image_encoder, self.sparse_scheduler_512, 
                            generator=generator, mode='sparse512', 
                            mc_threshold=mc_threshold, latent_index=latent_index, 
                            remove_interior=remove_interior, num_inference_steps=steps, guidance_scale=guidance_scale)[0]         
        return mesh   
        
    @torch.no_grad()
    def generate_dense(self, image, steps, guidance_scale, mc_threshold, seed):
        self.clear_memory()
        self.init_dense()
        
        generator=torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.prepare_image(image)
        
        mesh = self.inference(image, self.dense_vae, self.dense_dit, 
                            self.sparse_image_encoder, self.dense_scheduler, 
                            generator=generator, mode='dense', 
                            mc_threshold=mc_threshold, 
                            num_inference_steps=steps, guidance_scale=guidance_scale)[0]         
        return mesh    

    @torch.no_grad()
    def refine_dense_512(self, image, latent_index, steps, guidance_scale, remove_interior, mc_threshold, seed):
        self.clear_memory()
        self.init_sparse_512()    
            
        generator=torch.Generator(device=self.device).manual_seed(seed)
                
        latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size) 
        print(f"number of latent tokens: {len(latent_index)}")
        
        image = self.prepare_image(image)

        mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512, 
                            self.sparse_image_encoder, self.sparse_scheduler_512, 
                            generator=generator, mode='sparse512', 
                            mc_threshold=mc_threshold, latent_index=latent_index, 
                            remove_interior=remove_interior, num_inference_steps=steps, guidance_scale=guidance_scale)[0]         
        return mesh 

    @torch.no_grad()
    def refine_dense_1024(self, image, latent_index, steps, guidance_scale, remove_interior, mc_threshold, seed):
        self.clear_memory()
        self.init_sparse_1024()

        image = self.prepare_image(image)
            
        generator=torch.Generator(device=self.device).manual_seed(seed)

        latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size) 
        print(f"number of latent tokens: {len(latent_index)}")            
        
        mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024, 
                            self.sparse_image_encoder, self.sparse_scheduler_1024, 
                            generator=generator, mode='sparse1024', 
                            mc_threshold=mc_threshold, latent_index=latent_index, 
                            remove_interior=remove_interior, num_inference_steps=steps, guidance_scale=guidance_scale)[0]         
        return mesh     
    
    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, List[Image.Image]] = None,
        sdf_resolution: int = 1024,
        dense_sampler_params: dict = {'num_inference_steps': 50, 'guidance_scale': 7.0},
        sparse_512_sampler_params: dict = {'num_inference_steps': 30, 'guidance_scale': 7.0},
        sparse_1024_sampler_params: dict = {'num_inference_steps': 15, 'guidance_scale': 7.0},
        generator: Optional[torch.Generator] = None,
        remesh: bool = False,
        simplify_ratio: float = 0.95,
        mc_threshold: float = 0.2,
        remove_interior: bool = False,
        fill_holes: bool = False,
        remove_interior_512: bool = False,
        remove_interior_1024: bool = False,
        target_facenum: int = 200000,
        simplify_lib: str = "Pymeshlab"):

        image = self.prepare_image(image)
        
        #latent_index = self.inference(image, self.dense_vae, self.dense_dit, self.dense_image_encoder,self.dense_scheduler, generator=generator, mode='dense', mc_threshold=0.1, **dense_sampler_params)[0]
        
        #latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size)

        #512
        # mesh = Trimesh.load(f'C:\Git\Direct3D-S2\dwarf.obj',force='mesh')
        # mesh = normalize_mesh(mesh)
        # latent_index = mesh2index(mesh, size=512, factor=8)
        # latent_index = sort_block(latent_index, self.sparse_dit_512.selection_block_size)        

        # mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512, 
                                # self.sparse_image_encoder, self.sparse_scheduler_512, 
                                # generator=generator, mode='sparse512', 
                                # mc_threshold=mc_threshold, latent_index=latent_index, 
                                # remove_interior=remove_interior_512, **sparse_512_sampler_params)[0]
                           
        #1024
        mesh = Trimesh.load(f'C:\Git\Direct3D-S2\dwarf.obj',force='mesh')
        mesh = normalize_mesh(mesh)
        latent_index = mesh2index(mesh, size=1024, factor=8)
        latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size)        

        mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024, 
                            self.sparse_image_encoder, self.sparse_scheduler_1024, 
                            generator=generator, mode='sparse1024', 
                            mc_threshold=mc_threshold, latent_index=latent_index, 
                            remove_interior=remove_interior_1024, **sparse_1024_sampler_params)[0]                              
                                

        # latent_index = mesh2index(mesh, size=1024, factor=8)        
        # mesh = self.inference(image, self.sparse_vae_512, self.sparse_dit_512, 
                                # self.sparse_image_encoder, self.sparse_scheduler_512, 
                                # generator=generator, mode='sparse512', 
                                # mc_threshold=mc_threshold, latent_index=latent_index, 
                                # remove_interior=remove_interior_512, **sparse_512_sampler_params)[0]        
        
        torch.cuda.empty_cache()

        if sdf_resolution == 1024:
            del latent_index
            torch.cuda.empty_cache()
            mesh = normalize_mesh(mesh)
            latent_index = mesh2index(mesh, size=1024, factor=8)
            latent_index = sort_block(latent_index, self.sparse_dit_1024.selection_block_size)
            print(f"number of latent tokens: {len(latent_index)}")

            mesh = self.inference(image, self.sparse_vae_1024, self.sparse_dit_1024, 
                                self.sparse_image_encoder, self.sparse_scheduler_1024, 
                                generator=generator, mode='sparse1024', 
                                mc_threshold=mc_threshold, latent_index=latent_index, 
                                remove_interior=remove_interior_1024, **sparse_1024_sampler_params)[0]
            
        if remesh:
            if simplify_lib == "Pymeshlab":            
                from .utils.postprocessors import postprocessmesh
                mesh = postprocessmesh(mesh.vertices, mesh.faces, target_facenum)
            elif simplify_lib == "Meshlib":
                from .utils.meshlib import postprocessmesh
                mesh = postprocessmesh(mesh.vertices, mesh.faces, target_facenum)               
            # import trimesh
            # from direct3d_s2.utils import postprocess_mesh
            # filled_mesh = postprocess_mesh(
                # vertices=mesh.vertices,
                # faces=mesh.faces,
                # simplify=True,
                # simplify_ratio=simplify_ratio,
                # verbose=True,
                # fill_holes=fill_holes
            # )
            # mesh = trimesh.Trimesh(filled_mesh[0], filled_mesh[1])

        outputs = {"mesh": mesh}

        return outputs
        