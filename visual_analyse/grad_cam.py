# Author: Ahzyuan
# Date: 2024.12.02
# coding: utf-8

import argparse,os,re
import torch,cv2,pdb
import numpy as np
from rich import print
from time import strftime
from collections import defaultdict
from opengait.utils import config_loader

from .simple_model import SimpleModel
from .utils import cuda_dist, get_model_structure, temporal_align

'''
Usage:

# pwd: .../OpenGait
conda activate <your-env-for-opengait>
python -m visual_analyse.grad_cam -c configs/gaitset/gaitset.yaml 
'''

LAYER_FEAT = {} # used to store the output feature of layers to be analyzed
LAYER_GRAD = {} # used to store the gradient of layers to be analyzed
RES = {'imgs':[], # used to store the input silu imgs(in bgr mode) and the heatmaps of picked layers for each sequence
       'heatmaps':defaultdict(list)} 

def forward_hook(module, _, fea_out):  
    global LAYER_FEAT
    global model_name
    
    if 'gaitgl' in model_name.lower():
        fea_out = fea_out.permute(0,2,1,3,4) # B,C,T,H,W -> B,T,C,H,W
    
    if len(LAYER_FEAT[id(module)])>1: # adapt to the behavior of spliting in height dimension and re-concat
        fea_out = torch.cat([LAYER_FEAT[id(module)].pop(-1), fea_out],
                            dim=list(range(fea_out.ndim))[-2])
            
    LAYER_FEAT[id(module)].append(fea_out)
     
def backward_hook(module, _, grad_out): 
    global LAYER_GRAD
    global model_name
    
    if 'gaitgl' in model_name.lower():
        grad_out = grad_out[0].permute(0,2,1,3,4) # B,C,T,H,W -> B,T,C,H,W
    
    if len(LAYER_GRAD[id(module)])>1:
        grad_out = torch.cat([LAYER_GRAD[id(module)].pop(-1), grad_out[0]], 
                             dim=list(range(grad_out[0].ndim))[-2])

    else:
        grad_out = grad_out[0]
        
    LAYER_GRAD[id(module)].append(grad_out)

def hooking(pick_layer_idxs, idx_modules_map):    
    global LAYER_FEAT, LAYER_GRAD
    
    handles = []
    for layer_idx in pick_layer_idxs:  
        layer_name, layer = idx_modules_map[layer_idx]
        
        LAYER_FEAT[id(layer)] = [f'{layer_idx}_' + layer_name.replace('.', '-')]
        LAYER_GRAD[id(layer)] = [f'{layer_idx}_' + layer_name.replace('.', '-')]
        
        forward_handle = layer.register_forward_hook(hook=forward_hook)
        backward_handle = layer.register_full_backward_hook(hook=backward_hook)
        handles.append(forward_handle)
        handles.append(backward_handle)
        
        print(f'[green][+] Hooking layer: {layer_name}[/]')
    return handles

def dump_result(seq_frames, save_dir:str, model_name:str, seq_name:str):
    """
    Save everything in `RES` to `args.save_dir/<model_name>_<seq_name>.npz`.
    the `npz` file includes the so called `everything` bellow:
    - `imgs`: ndarray, shape is `(T,H,W,3)`; all of the silu images(bgr mode) in a sequence.
    - `heatmaps`: dict{layer_name: ndarray, shape is `(T,H,W,3)`}; heatmaps for each picked layer per image.

    Args:
        seq_frames (np.ndarray): all of the silu images(bgr mode) in a sequence, shape is `(T,H,W,3)`.
        save_dir (str): save directory.
        model_name (str): model name
        seq_name (str): sequence name in formmat of `<id>-<type>-<view>`
    """
    global RES

    RES['imgs'] = seq_frames # ndarray, (T,H,W,3)
    RES['heatmaps'] = {k:np.stack(v, axis=0) for k,v in RES['heatmaps'].items()} # {layer_name: ndarray(T,H,W,3)}
    
    save_path = os.path.join(save_dir, f'{model_name}_{seq_name}' )
    np.savez_compressed(save_path, **RES)
    
    RES = {'imgs':[], 
           'heatmaps':defaultdict(list)} 

def gallery_inference_hook(func, gallery_data): 
    """
    Used to collect inference features of gallery samples.

    Args:
        func (function handle): `SimpleModel.inference_step`
        gallery_data (list): container for gallery features
    """
    def wrapper(ipts, pick_key):
        # ipts: transformed_data, labels, types, views, sequence_length(int)
        final_feat, img, pick_key = func(ipts, pick_key) # img: T,1,H,W
        final_feat = final_feat.detach()
        gallery_data.append(final_feat)
        return None, None, pick_key

    return wrapper

def pre_inference_hook(func, args, model):
    """
    Decorator for `SimpleModel.inference_step`.
    Use a gait sequence as the model input, and draw Grad-CAM heatmap on the middle frame.

    Args:
        func (function handle): `SimpleModel.inference_step`
        args (namespace): arguments passed to this script
        model (SimpleModel): the specific model defined in config file
    """

    def wrapper(ipts, pick_key):
        global LAYER_FEAT, LAYER_GRAD
        global gallery_data
        
        # ipts: transformed_data, labels, types, views, sequence_length(int)
        final_feat, img, pick_key = func(ipts, pick_key) # img: T,1,H,W
        bgr_imgs = torch.repeat_interleave(img, 3, dim=1) # gray -> bgr
        bgr_imgs = bgr_imgs.permute(0,2,3,1).cpu().numpy() # T,3,H,W -> T,H,W,3
        bgr_imgs *= 255
        
        # use minimum distance for backward to get gradient of the wanted layer
        model.zero_grad()
        dist = cuda_dist(final_feat, gallery_data) # euc dist
        pred = torch.min(dist)
        pred.backward() 

        for layer_id, (layer_name, layer_feats) in LAYER_FEAT.items():
            # layer_feat & layer_grad: [(B=1), T, C, H, W]
            layer_grads = LAYER_GRAD[layer_id][-1]
            
            layer_feats = torch.squeeze(layer_feats)
            layer_grads = torch.squeeze(layer_grads)
            
            if layer_feats.ndim == 3: # features after set-level operations
                layer_feats = torch.stack([layer_feats]*len(bgr_imgs), dim=0)
                layer_grads = torch.stack([layer_grads]*len(bgr_imgs), dim=0)
            
            # situation that layer compressed the temporal dimension, e.g. gaitgl.GLconv
            if len(bgr_imgs) != len(layer_feats): 
                compresses_imgs = temporal_align(bgr_imgs, layer_feats)
            else:
                compresses_imgs = bgr_imgs
            
            # draw heatmap for input sequence
            for bgr_img, layer_feat, layer_grad in zip(compresses_imgs, layer_feats, layer_grads):
                main(bgr_img, layer_name, layer_feat, layer_grad)
        
        LAYER_FEAT = {k:v[:1] for k,v in LAYER_FEAT.items()}
        LAYER_GRAD = {k:v[:1] for k,v in LAYER_GRAD.items()}
        
        dump_result(seq_frames=bgr_imgs,
                    save_dir=args.save_dir, 
                    model_name=args.cfgs['model_cfg']['model'],
                    seq_name=f'{ipts[1][0]}-{ipts[2][0]}-{ipts[3][0]}')
        
        return final_feat, img, pick_key

    return wrapper

def main(bgr_img, layer_name, layer_feat, layer_grad):
    global RES

    layer_feat = layer_feat.detach().cpu().numpy() # [C, H, W]
    pooled_grads = torch.mean(layer_grad, dim=list(range(layer_grad.ndim))[-2:], keepdim=True) # GAP for last two dimensions
    pooled_grads = pooled_grads.detach().cpu().numpy()
    
    heatmap = layer_feat * pooled_grads # [C, H, W]
    heatmap = np.mean(heatmap, axis=0)
    
    heatmap = abs(heatmap) # strange, most of the features and gradient are negative
    minimax_val = np.ptp(heatmap)
    minimax_val = minimax_val if minimax_val > 0 else 1 # avoid zero division
    heatmap = (heatmap - np.min(heatmap)) / np.ptp(heatmap)
    heatmap = np.uint8(255 * heatmap) 
    heatmap = cv2.resize(heatmap, (bgr_img.shape[1], bgr_img.shape[0]))
    
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # merge = heatmap * args.gradcam_opacity + bgr_img*(1-args.gradcam_opacity)
    
    # push heatmap to stack in `RES`
    RES['heatmaps'][layer_name].append(heatmap)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='Path of the config file.')
    parser.add_argument('-p', '--probe_dir', type=str, default='visual_analyse/Probes',
                        help='Path of the probe data which will be analysed. Note that the structure of the dir should be the same as a dataset. \
                              Default: visual_analyse/Probes')
    parser.add_argument('-g', '--gallery_dir', type=str, default='visual_analyse/Gallery',
                        help='Path of the gallery data. Note that the structure of the dir should be the same as a dataset. \
                              Default: visual_analyse/Gallery')
    parser.add_argument('-s', '--save_dir', type=str, default=f"visual_analyse/Results/Grad_CAM/{strftime('%Y-%m-%dT%H-%M-%S')}",
                        help='Path to save the results. Default: visual_analyse/Results/Grad_CAM/<time>')
    parser.add_argument('--device',  type=int, default=0,
                        help='GPU index to use. Default: 0')
    args = parser.parse_args()
    args.config_path = os.path.abspath(args.config_path)
    args.probe_dir = os.path.abspath(args.probe_dir)
    args.gallery_dir = os.path.abspath(args.gallery_dir)
    args.save_dir = os.path.abspath(args.save_dir)
    assert os.path.exists(args.config_path), f"{args.config_path} does not exist."
    assert os.path.exists(args.probe_dir), f"{args.probe_dir} not found!"
    assert os.path.exists(args.gallery_dir), f"{args.gallery_dir} not found!"
    assert torch.cuda.device_count() > args.device >= 0, f"GPU device `{args.device}` not found! \
        Use `{','.join([str(i) for i in range(torch.cuda.device_count())])}` instead."
    os.makedirs(args.save_dir, exist_ok=True)
    
    # load config
    ## Note: you should assure that the `evaluator_cfg` field and `data_cfg` field are well specified
    cfgs = config_loader(args.config_path)
    cfgs['evaluator_cfg']['sampler']['batch_size'] = 1 # DON'T CHANGE!!!!
    cfgs['data_cfg']['dataset_root'] = args.gallery_dir
    args.cfgs = cfgs
    
    # build model according to the specified checkpoint defined in config
    model = SimpleModel(args)
    model_name = cfgs['model_cfg']['model']
    origin_inference_step = model.inference_step
    model.eval()
    
    # get features of gallery data in `args.gallery_dir`
    gallery_data = []
    model.inference_step = gallery_inference_hook(model.inference_step, gallery_data)
    print('[bold green]Extracting gallery data...[/]')
    model.inference()
    gallery_data = torch.cat(gallery_data, dim=0) # N,C,P
    
    # change the data exposed to model to the `args.probe_dir`
    cfgs['data_cfg']['dataset_root'] = args.probe_dir
    model.samples_loader = model.get_loader(cfgs['data_cfg'], train=False)
    model.inference_step = pre_inference_hook(origin_inference_step, args, model)
    
    # display modules info recursively and pick the layers to draw Grad-CAM
    idx_modules_map = get_model_structure(model, model_name)
    
    print('[bold green]Pick the layer numbers to draw Grad_CAM[/]')
    print('[magenta](You can pick multiple layers, seperate by space, e.q. `10 15`)[/]')
    pick_idxs = list(
                    map(int, 
                        re.split("[^\d]+", 
                                 input('Enter the layers\' numbers: ').strip()
                        )
                    )
                )
    for idx in pick_idxs:
        if idx not in idx_modules_map:
            raise ValueError(f'{idx} is invalid.')
    pick_idxs = list(set(pick_idxs)) # remove duplicates
    pick_idxs.sort() # necessary, the order of the layers determines the order of the `RES['heatmaps]`
    
    # hook the desired layers
    handles = hooking(pick_idxs, idx_modules_map)
    
    # forward propagation and backward propagation(defined in `main`)
    model.inference()
    
    # clean up hooks
    list(map(lambda x:x.remove(), handles))
    
    print(f'[bold][green]Results saved to[/green] [underline magenta]{args.save_dir}[/]')
    print(f'[bold][dark_goldenrod]\nVisualization:\n\npython -m visual_analyse.visualizer visual_gradcam [magenta]{args.save_dir}[/magenta] -r 5\n[/]')
    print('[bold][grey54]more info see: [underline]python -m visual_analyse.visualizer visual_gradcam --help[/]')