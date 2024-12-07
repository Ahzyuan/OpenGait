# Author: Ahzyuan
# Date: 2024.12.02
# coding: utf-8

import os,time
import torch
import numpy as np
import torch.nn.functional as F
from typing import Union
from rich import box
from rich.live import Live
from rich.table import Table
from rich.console import Console

def cuda_dist(x, y, metric='euc'):
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

def fake_gallery_data(probe_data, num=10): 
    # probe_data: [1, C, P]
    forge_mean = torch.mean(probe_data) 
    forge_std = torch.std(probe_data)  
    
    forge_gallery = [probe_data * (1 + torch.normal(forge_mean, forge_std, (1,),device=probe_data.device)) 
                     for _ in range(num)]
    forge_gallery = torch.cat(forge_gallery, dim=0)

    return forge_gallery # N,C,P

def get_model_structure(model, model_name='Model', save_dir=None):
    console = Console()
    table = Table(title=model_name, show_lines=True, box=box.ASCII)
    table.add_column("Id", justify="center", style="green", no_wrap=True)
    table.add_column("Name", justify="center", style="cyan")
    table.add_column("Module", justify="center", style="gold3")
    
    idx_modules_map = {}
    with Live(table, console=console, auto_refresh=True, vertical_overflow='visible'): 
        idx = 1
        for name, module in model.named_modules():
            if not module._modules:
                table.add_row(str(idx), name, str(module))
                idx_modules_map[idx] = (name, module)
                idx += 1
                time.sleep(0.25)
        
    if save_dir:
        save_path = os.path.join(save_dir, 'model_structure.txt')   
        with open(save_path,'w',encoding='utf-8') as stc_writer:  
            for name,module in model.named_children():
                stc_writer.write(name+' : '+str(module)+'\n\n')  
    
    return idx_modules_map # {idx(int): (module_name(str), module(nn.Module))}

def temporal_align(sr_mat:Union[torch.Tensor,np.ndarray], 
                   des_mat:Union[torch.Tensor,np.ndarray]):
    """
    Align the sr_mat to the des_mat in temporal dimension.

    Args:
        sr_mat (Union[torch.Tensor,np.ndarray]): the source matrix to be aligned. Supposed the first dimension is temporal.
        des_mat (Union[torch.Tensor,np.ndarray]): the destination matrix to align to. Supposed the first dimension is temporal.

    Returns:
        np.ndarray: the aligned sr_mat
    """
    if isinstance(sr_mat, np.ndarray):
        sr_mat = torch.from_numpy(sr_mat)
    
    sr_mat = sr_mat.to(dtype=torch.float32)

    aligned_mat = sr_mat.permute(1,0,2,3)
    aligned_mat = F.adaptive_avg_pool3d(aligned_mat, (des_mat.shape[0], *sr_mat.shape[2:]))
    aligned_mat = aligned_mat.permute(1,0,2,3)
    return aligned_mat.numpy()