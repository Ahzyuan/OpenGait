# Author: Ahzyuan
# Date: 2024.12.16
# coding: utf-8

import argparse,os,pdb
import torch
import numpy as np
from rich import print
from time import strftime
from opengait.utils import config_loader

from .simple_model import SimpleModel

'''
Usage:

# pwd: .../OpenGait
conda activate <your-env-for-opengait>
python -m visual_analyse.tsne -c configs/gaitset/gaitset.yaml 
'''

def pre_inference_hook(func, labels_map): 
    """
    Decorator for `SimpleModel.inference_step`.
    Used to collect inference features of input samples.

    Args:
        - func (function handle): `SimpleModel.inference_step`

        - labels_map (dict): map from label to its index in the dataset
    
    Returns:
        the decorated function
    """
    def wrapper(ipts, pick_key):
        global feats, labels

        # ipts: transformed_data, labels, types, views, sequence_length(int)
        final_feat, img, pick_key = func(ipts, pick_key) # final_feat: B,C,P
        final_feat = final_feat.detach()
        
        feats.append(final_feat.reshape(final_feat.shape[0],-1)) # Reshape to [B,C*P] for t-SNE's dimensionality reduction.
        labels.append(labels_map[ipts[1]]) # list[str]
        return None, None, pick_key

    return wrapper
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='Path of the config file.')
    parser.add_argument('-d', '--dataset_dir', type=str, default=None,
                        help='Path of the pkl dataset which will be analysed. Default: the value of `dataset_root` in config.')
    parser.add_argument('-s', '--save_dir', type=str, default=f"visual_analyse/Results/TSNE/{strftime('%Y-%m-%dT%H-%M-%S')}",
                        help='Path to save the results. Default: visual_analyse/Results/TSNE/<time>')
    parser.add_argument('--device',  type=int, default=0,
                        help='GPU index to use. Default: 0')
    args = parser.parse_args()
    args.config_path = os.path.abspath(args.config_path)
    args.save_dir = os.path.abspath(args.save_dir)
    assert os.path.exists(args.config_path), f"{args.config_path} does not exist."
    assert torch.cuda.device_count() > args.device >= 0, f"GPU device `{args.device}` not found! \
        Use `{','.join([str(i) for i in range(torch.cuda.device_count())])}` instead."
    os.makedirs(args.save_dir, exist_ok=True)
    
    # load config
    ## Note: you should assure that the `evaluator_cfg` field and `data_cfg` field are well specified
    cfgs = config_loader(args.config_path)
    model_name = cfgs['model_cfg']['model']
    set_name = cfgs['data_cfg']['dataset_name']

    args.dataset_dir = args.dataset_dir or cfgs['data_cfg']['dataset_root']
    args.dataset_dir = os.path.abspath(args.dataset_dir)
    cfgs['data_cfg']['dataset_root'] = args.dataset_dir
    assert os.path.exists(args.dataset_dir), f"{args.dataset_dir} not found!"    

    cfgs['evaluator_cfg']['sampler']['batch_size'] = 1 # DON'T CHANGE!!!!
    args.cfgs = cfgs

    # build model according to the specified checkpoint defined in config
    model = SimpleModel(args)
    model.eval()

    # get features of gallery data in `args.gallery_dir`
    feats, labels = [], []
    model.inference_step = pre_inference_hook(model.inference_step, labels_map=model.dataset.label_set)
    print('[bold green]Extracting features...[/]')
    model.inference()
    feats = torch.cat(feats, dim=0) # N,C,P

    # save features and corresponding labels to npz
    save_path = os.path.join(args.save_dir, f'{model_name}_{set_name}.npz')
    np.savez_compressed(save_path,
                        feats=feats.cpu().numpy(),
                        labels=np.array(labels))

    print(f'[bold][green]Results saved to[/green] [underline magenta]{save_path}[/]')
    print(f'[bold][dark_goldenrod]\nVisualization:\n\npython -m visual_analyse.visualizer visual_tsne [magenta]{save_path}[/magenta]\n[/]')
    print('[bold][grey54]more info see: [underline]python -m visual_analyse.visualizer visual_tsne --help[/]')