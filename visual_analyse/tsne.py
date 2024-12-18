# Author: Ahzyuan
# Date: 2024.12.16
# coding: utf-8

import argparse,os,traceback,warnings,pdb
import torch
import numpy as np
from rich import print
from queue import Queue
from time import strftime
from threading import Thread
from sklearn.manifold import TSNE
from opengait.utils import config_loader

from .simple_model import SimpleModel

'''
Usage:

# pwd: .../OpenGait
conda activate <your-env-for-opengait>
python -m visual_analyse.tsne -c configs/gaitset/gaitset.yaml 
'''

warnings.filterwarnings("ignore", category=FutureWarning)

def pre_inference_hook(func, labels_map): 
    """
    Decorator for `SimpleModel.inference_step`.
    Used to collect inference features of input samples.

    Args:
        - `func` (function handle): `SimpleModel.inference_step`

        - `labels_map` (dict): map from label to its index in the dataset
    
    Returns:
        the decorated function
    """
    def wrapper(ipts, pick_key):
        global msg_queue

        # ipts: transformed_data, labels, types, views, sequence_length(int)
        final_feat, img, pick_key = func(ipts, pick_key) # final_feat: 1,C,P
        final_feat = final_feat.detach().cpu().numpy()
        
        # Reshape to [1,C*P] for t-SNE's dimensionality requirement
        msg_queue.put((np.reshape(final_feat, (final_feat.shape[0],-1), order='C'),
                      labels_map[ipts[1]]))# list[str]

        return None, None, pick_key

    return wrapper

def main(feat:np.ndarray, args:argparse.Namespace):
    """
    Perform t-SNE on the given features with passing-in arguments.

    Args:
        - `feat` (np.ndarray): reshaped inference feature, shape is [1,C*P] 
        - `args` (argparse.Namespace): pass in arguments

    Returns:
        np.ndarray : feature after t-SNE embedding
    """

    # print customized t-sne args
    if args.TSNE_ARGS:
        args.TSNE_ARGS['verbose'] = args.verbose
        print('[green]' + '-'*30 + '[/]')
        print('[green]Receive args for t-sne:[/]')
        for k,v in args.TSNE_ARGS.items():
            print(f'[green]{k}: {v}[/]')
        print('[green]' + '-'*30 + '[/]')

    # perform t-SNE
    if args.TSNE_ARGS:
        tsne = TSNE(**args.TSNE_ARGS)
    else:
        tsne = TSNE(n_components=args.embed_dim, 
                    init='pca', 
                    method='exact',
                    n_jobs=-1,
                    n_iter=1000, 
                    learning_rate='auto',
                    random_state=args.seed,
                    verbose=args.verbose)

    embedded_feat = tsne.fit_transform(feat) # N, embed_dim

    # dimension post-processing 
    if args.embed_dim > 3:
        print(f'[bold yellow]⚠ Embedding dim is set to {args.embed_dim}, only use the [red]first 3 dims[/red] for visualization![/]')
        embedded_feat = embedded_feat[:,:3] # N, 3
            
    if args.embed_dim < 2:
        print(f'[bold yellow]⚠ Embedding dim is set to {args.embed_dim}, will be zero-padded to two dimensions for visualization![/]')
        embedded_feat = np.c_[embedded_feat, np.zeros(embedded_feat.shape[0])] # N, 2

    return embedded_feat

def msg_inference(func):
    """
    Decorator for `SimpleModel.inference`.
    Used to notify the tsne_thread that inference has ended.

    Args:
        - `func` (function handle): `SimpleModel.inference`
    """
    def wrapper(*args, **kwargs):
        global msg_queue
        func()
        msg_queue.put((None, None))
    return wrapper

def bucket_tsne(args:argparse.Namespace):
    """
    Perform t-SNE on the given features in a bucket-wise manner.

    Args:
        - `args` (argparse.Namespace): pass in arguments

    Returns:
        None
    """
    global msg_queue
    global feats, labels

    bucket_feats = []

    while True:
        # reshape_feat: [1, C*P], label: str
        reshaped_feat, label = msg_queue.get()

        # inference end
        if reshaped_feat is None:
            # perform tsne on the remaining features
            if bucket_feats:
                cat_feats = np.concatenate(bucket_feats, axis=0)
                feats.append(main(cat_feats, args))
            break

        bucket_feats.append(reshaped_feat)
        labels.append(label)

        if len(bucket_feats) == args.bucket_size:
            cat_feats = np.concatenate(bucket_feats, axis=0)
            feats.append(main(cat_feats, args))
            bucket_feats = []
            del cat_feats
    
    # feats: [N, embed_dim]
    feats = np.concatenate(feats, axis=0)
    assert len(feats) == len(labels), \
        f'[bold red]Error: length of feats ({len(feats)}) does not match length of labels ({len(labels)}).[/]'
    
    # normalize the embedding to [0,1]
    x_min, x_max = np.min(feats, 0), np.max(feats, 0)
    feats = (feats - x_min) / (x_max - x_min)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='Path of the config file.')
    parser.add_argument('-e', '--embed_dim', type=int, default=2,
                        help='Embedding dim of t-sne. Defaults to 2.')
    parser.add_argument('-b', '--bucket_size', type=int, default=2e4,
                        help='Size of a bucket containing features to be performed t-sne. This argument is memory-friendly. Default: 20000')
    parser.add_argument('-d', '--dataset_dir', type=str, default=None,
                        help='Path of the pkl dataset which will be analysed. Default: the value of `dataset_root` in config.')
    parser.add_argument('-s', '--save_dir', type=str, default=f"visual_analyse/Results/TSNE/{strftime('%Y-%m-%dT%H-%M-%S')}",
                        help='Path to save the results. Default: visual_analyse/Results/TSNE/<time>')
    parser.add_argument('-T', '--TSNE_ARGS', type=dict, default={},
                        help='Used to receive customized TSNE drawing parameters. Defaults to {}.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to print the progress while embedding. Defaults to False.')
    parser.add_argument('--seed',  type=int, default=1597,
                        help='Random seed, used to reproduce the same embedding. Defaults to 1597.')
    parser.add_argument('--device',  type=int, default=0,
                        help='GPU index to use. Default: 0')
    args = parser.parse_args()
    args.config_path = os.path.abspath(args.config_path)
    args.save_dir = os.path.abspath(args.save_dir)
    assert os.path.exists(args.config_path), f"{args.config_path} does not exist."
    assert args.embed_dim > 0, f'Target dim should be positive, but got {args.embed_dim}!'
    assert args.bucket_size > 0, f'Bucket size should be positive, but got {args.bucket_size}!'
    assert args.seed > 0, f'Bucket size should be positive, but got {args.seed}!'
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
    model.inference_step = pre_inference_hook(model.inference_step, labels_map=model.dataset.label_set)
    model.inference = msg_inference(model.inference)
    model.eval()

    # bucket number warning
    samples_num = len(model.dataset)
    buckets_num = (samples_num-1) // args.bucket_size + 1 # ceil division
    if buckets_num >= 3:
        print(f'[bold yellow]●  Warning: All samples\' features will be divided into {buckets_num} parts when `-b` or `--bucket_size` is set to {args.bucket_size}[/]')
        print('[bold yellow]   T-SNE will be performed on each part independently and [red]may not be accurate.[/]')
        print('[bold yellow]   If there is enough memory, please increase `-b` or `--bucket_size` when drawing T-SNE.[/]')
        print('[bold blue]   Continue? Press [green]y[/green] or [red]n[/]')
        go_ahead = input()
        if go_ahead.lower() != 'y':
            print('[bold red]Exit ...[/]')
            exit()

    # create thread for inference and tsne
    msg_queue = Queue()
    feats, labels = [], []
    infer_thread = Thread(target=model.inference)
    tsne_thread = Thread(target=bucket_tsne, args=(args,))

    # inference and tsne
    print('[bold green]Extracting features...[/]')

    try:
        tsne_thread.start()
        infer_thread.start()

        tsne_thread.join()
        infer_thread.join()

        # save features and corresponding labels to npz
        save_path = os.path.join(args.save_dir, f'{model_name}_{set_name}.npz')
        np.savez_compressed(save_path,
                            feats=feats,
                            labels=np.array(labels))

        print(f'[bold][green]Results saved to[/green] [underline magenta]{save_path}[/]')
        print(f'[bold][dark_goldenrod]\nVisualization:\n\npython -m visual_analyse.visualizer visual_tsne [magenta]{save_path}[/magenta]\n[/]')
    except MemoryError:
        print(traceback.format_exc())
        print(f'[bold yellow]The data volume in {args.dataset_dir} is too large.[/]')
        print(f'[bold yellow]Got bucket_size is set to {args.bucket_size}. You can pass in a smaller `bucket_size` to prevent memory overflow: \n[/]')
        print('[bold grey54]    python -m visual_analyse.visualizer visual_tsne -c <config_path> -b <bucket_size> \n[/]')
    
    print('[bold][grey54]more info see: [underline]python -m visual_analyse.visualizer visual_tsne --help[/]')