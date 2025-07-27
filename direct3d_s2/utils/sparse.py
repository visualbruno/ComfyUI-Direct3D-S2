import torch
import numpy as np

def sort_block(latent_index, block_size):
    device = latent_index.device
    latent_index_block = latent_index.cpu().numpy()
    latent_index_block[..., 1:] = latent_index_block[..., 1:] // block_size
    latent_index_inblock = latent_index.cpu().numpy()
    latent_index_inblock[..., 1:] = latent_index_inblock[..., 1:] % block_size
    sort_index = np.lexsort((
        latent_index_inblock[..., 3],
        latent_index_inblock[..., 2],
        latent_index_inblock[..., 1],
        latent_index_block[..., 3],
        latent_index_block[..., 2],
        latent_index_block[..., 1])
    )
    sort_index = torch.from_numpy(sort_index).to(device)
    return latent_index[sort_index]

def extract_tokens_and_coords(conditions, token_mask, num_cls=1, num_reg=4):
    device = conditions.device
    B = conditions.size(0)
    patch_size = token_mask.size(1) 
    
    class_tokens = conditions[:, 0:num_cls, :]  # [B, 1, 1024]
    register_tokens = conditions[:, num_cls:num_cls+num_reg, :]  # [B, 4, 1024]
    patch_tokens = conditions[:, num_cls+num_reg:, :]  # [B, 1369, 1024]

    selected_tokens_list = []
    coords_list = []

    for batch_idx in range(B):
        cls_tokens = class_tokens[batch_idx]  # [1, 1024]
        reg_tokens = register_tokens[batch_idx]  # [4, 1024]
        cls_reg_tokens = torch.cat([cls_tokens, reg_tokens], dim=0)  # [5, 1024]

        cls_coord = torch.tensor([[batch_idx, 0, 0, 1]] * num_cls, device=device)
        reg_coords = torch.tensor([[batch_idx, 0, 0, 1]] * num_reg, device=device)
        cls_reg_coords = torch.cat([cls_coord, reg_coords], dim=0)  

        mask = token_mask[batch_idx] 
        pos = mask.nonzero(as_tuple=False) 
        K = pos.size(0)

        if K > 0:
            h, w = pos[:, 0], pos[:, 1]
            indices = h * patch_size + w  #
            patches = patch_tokens[batch_idx][indices] 

            batch_ids = torch.full((K, 1), batch_idx, device=device)
            x = w.unsqueeze(1)  
            y = h.unsqueeze(1)  
            patch_coords = torch.cat([batch_ids, x, y, torch.zeros((K, 1), device=device)], dim=1)

            combined_tokens = torch.cat([cls_reg_tokens, patches], dim=0)
            combined_coords = torch.cat([cls_reg_coords, patch_coords], dim=0)
        else:
            combined_tokens = cls_reg_tokens
            combined_coords = cls_reg_coords

        selected_tokens_list.append(combined_tokens)
        coords_list.append(combined_coords)

    selected_tokens = torch.cat(selected_tokens_list, dim=0)
    coords = torch.cat(coords_list, dim=0)
    
    return selected_tokens, coords