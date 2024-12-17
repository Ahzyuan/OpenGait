# Author: Ahzyuan
# Date: 2024.12.02
# coding: utf-8

import os,cv2
import fire,pdb
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from rich import print
from rich.box import HEAVY
from rich.text import Text
from rich.panel import Panel
from typing import Union,Tuple
from rich.console import Console
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import temporal_align

'''
Usage:

# pwd: .../OpenGait
conda activate <your-env-for-opengait>

## draw gradcam
python -m visual_analyse.visualizer visual_gradcam \
<path-to-npz-file or dir> \
[-s <path-to-save-dir>] \
[-h <heatmap_opacity>] \
[-d <draw_mode>] \
[-r <resize_ratio>] \
[-f <font_scale>]

## draw t-sne
python -m visual_analyse.visualizer visual_tsne \
<path-to-npz-file or dir> \
[-e <embed_dim>] \
[-m <marker_size>] \
[-t <transparent>] \
[-d <draw_mode>] \
[-v <verbose or not>] \
[-T <dict of customized t-sne args>] \
[--seed <random_seed>] \
[--save_path <save_path>] \
[--save_ext <save_extention>]
'''

class Visualizer:
    def dash_print(self, string:str):
        """
        Print string with dash in the middle of the terminal

        Args:
            string (str): strings to be printed
            
        Returns:
            None
        """
        console = Console()
        dash_num = (console.width - len(string))//2 -1
        console.print('-'*dash_num + string + '-'*dash_num , 
                      style='bold green',
                      justify='center')

    def plot_embedding(self,
                       feats:np.ndarray, labels:Union[list,np.ndarray],
                       fig_size:Tuple[float,float]=(8.0,8.0), marker_size:int = 50,
                       cmap:str='Spectral',
                       title:Union[str, None]=None):
        """
        Plot t-sne embedding point in 2D or 3D space.

        Args:
            - feats (np.ndarray): t-sne embedding features.

            - labels (Union[list,np.ndarray]): labels corresponding to each feature, it is a vertor with same length as `feats`

            - fig_size (Tuple[float,float], optional): figure size in inches. Defaults to (8.0,8.0).

            - marker_size (int, optional): size of scatter. Defaults to 50.

            - cmap (str, optional): colormap. Defaults to 'Spectral'.

            - `title` (Union[str, None], optional): title of the plot. Defaults to None.

        Returns:
            a `matplotlib.axes._subplots.AxesSubplot` object containing the plot.
        """
        cls_set = list(set(labels)) if isinstance(labels, list) else np.unique(labels).tolist()
        cls_set.sort()
        cls_idx_map = {cls:idx for idx, cls in enumerate(cls_set)}
        cls_idx = np.array([cls_idx_map[cls] for cls in labels])

        if feats.shape[1] == 3:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection='3d')
            
            plot_3d = True
        else:
            fig, ax = plt.subplots(1, 1, figsize=fig_size)

            if feats.shape[1] == 1:
                feats = np.c_[feats, np.zeros(feats.shape[0])]
            
            plot_3d = False

        feats = feats.T # 2(3), N

        # draw scatter
        im = ax.scatter(*feats, # position
                        s=marker_size, alpha=1.0, edgecolors='none', linewidths=marker_size/20.0, # shape, opacity, and edge color of markers
                        c=cls_idx, cmap=cmap, # color
                        vmin=0, vmax=len(cls_set)-1) # color map range
        
        # draw color mapping bar
        aspect = 30 # color bar ratio between height and width
        cls_num = len(cls_set)

        if plot_3d:
            position_args = {'anchor':(0.5, 1.8),
                             'orientation':'horizontal',
                             'shrink':0.5}
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.%", pad=0.05)
            position_args = {'cax':cax,
                             'orientation':'vertical',
                             'shrink':1,
                             'aspect':aspect}
            
        cbar = fig.colorbar(im,
                            ax=ax,
                            boundaries=np.arange(cls_num + 1) - 0.5,
                            **position_args)
        
        # ticks management
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        cbar.set_ticks(np.arange(cls_num))
        if plot_3d:
            cbar.ax.set_xticklabels(cls_set, rotation=45)
            ax.zaxis.set_visible(False)
        else:
            cbar.ax.set_yticklabels(cls_set)

        # set title
        if title:
            ax.set_title(title, fontsize=3*fig_size[0])

        # tight layout
        plt.tight_layout()
        
        return ax        

    def adjust_3d(self, 
                  ax_pkl:str, 
                  save_path:Union[str, None]=None, save_ext:str='png', 
                  transparent:bool=True):
        """
        Adjust perspective of 3D-embedding plot 

        Args:
            - `ax_pkl` (str): path of the 3D-embedding pkl file, which contains a `matplotlib.axes._axes.Axes` object.

            - `save_path` (Union[str, None], optional): path to save adjusted image. If it's set to None, 
                                                        then will save result in the same directory as the `ax_pkl` in specified `save_ext` format.
                                                        Defaults to None.

            - `save_ext` (str, optional): extension of the saved image, \  
                                          support ['png','jpg','jpeg','bmp','tiff','tif','svg']. \  
                                          Defaults to 'png'. . Defaults to 'png'.

            - `transparent` (bool, optional): Whether to save transparent image. Default to True.
        
            Returns:
                None
        """
        # validate saving extention
        save_ext = save_ext.lower().replace('.','')
        if save_ext not in ['png','jpg','jpeg','bmp','tiff','tif','svg']:
            print(f'[bold red]Saving as `{save_ext}` file is not supported![/]')
            print('[bold dark_goldenrod]Alternativly, you can use `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `svg`[/]')
            print('[bold dark_goldenrod]Choose one and pass to `--save_ext` when using [underline]python -m visual_analyse.visualizer visual_tsne`[/]')
            return
        
        # load ax pkl file
        with open(os.path.abspath(ax_pkl), 'rb') as f:
            ax = pk.load(f)

        # save changes
        try:
            plt.show()
            save_path = save_path or ax_pkl.replace('pkl', save_ext)
            save_path = os.path.abspath(save_path)
            ax.figure.savefig(save_path, 
                              bbox_inches='tight', 
                              dpi=300, 
                              transparent=transparent)
            print(f'[bold green]Changes saved to [magenta]{save_path}[/]')
        except:
            print('[bold red]pkl file broken![/]')

    def visual_gradcam(self, 
                       npz_path:str, 
                       save_dir:Union[str,None] = None, 
                       heatmap_opacity:float = 0.6,
                       draw_mode:str = 'merge',
                       resize_ratio:float = 1.0,
                       font_scale:float = 2.5):
        """
        Merge Grad-CAM heatmap with original image and save to disk.

        Args:
            - `npz_path` (str): path of a npz file or path of the directory containing npz files.
            
            - `save_dir` (Union[str,None], optional): path of a directory to save result images. When it's set to None,
                will be save to the same directory as the npz file with the npz file's name.Defaults to None.
                
            - `heatmap_opacity` (float, optional): opacity of the heatmap. Defaults to 0.6.
            
            - `draw_mode` (str, optional): the mode to draw heatmap, can be 'merge', 'compare' or 'both'. When it's set to 'merge',
                will only save each heatmap as a separate image in the subdirectory with corresponding layer name. When it's set
                to 'compare', will draw origin image as well as all different layers' heatmaps in a figure. When it's set to 'both',
                will perform the behavior that 'merge' and 'compare' will do, respectively. Defaults to 'merge'.
                
            - `resize_ratio` (float, optional): image resize ratio. Defaults to 1.0.
            
            - `font_scale` (float, optional): font size scale ratio. This parameter only takes effect on 'compare' mode. Defaults to 2.5.
        
        Returns:
            None
        """
        assert heatmap_opacity >= 0 and heatmap_opacity <= 1, f'heatmap_opacity should be in range [0,1], but got {heatmap_opacity}!'
        assert draw_mode in ['merge', 'compare', 'both'], f'draw_mode should be in [merge, compare, both], but got {draw_mode}!'
        
        # recursivly draw the heatmap for all npz files in the given `npz_path` and sub-directories
        if os.path.isdir(npz_path):
            npz_paths = glob(npz_path + '/**/*.npz',recursive=True)
            if not npz_paths:
                print(f'[bold red][underline]{npz_path}[/underline] does not contain any npz file![/]')
                return
            
            for npz_path in npz_paths:
                self.visual_gradcam(npz_path=npz_path, 
                                    save_dir=save_dir,
                                    heatmap_opacity=heatmap_opacity,
                                    draw_mode=draw_mode,
                                    resize_ratio=resize_ratio)
            return
        
        # validate file extention
        if not npz_path.endswith('.npz'):
            print(f'[bold red][underline]{npz_path}[/underline] is not a npz file![/]')
            return

        # load imgs and layers' heatmaps stored in npz file
        npz_path = os.path.abspath(npz_path)
        with np.load(npz_path, allow_pickle=True) as datas:
            imgs = datas['imgs'].astype('uint8') # T,H,W,3
            heatmaps = datas['heatmaps'].tolist() # {layer_name: ndarray(T,H,W,3)}
        draw_dict = {'origin':imgs} 
        draw_dict.update(heatmaps) # {layer_name: ndarray(T,H,W,3)}
        del heatmaps
        
        # save merged results or comparation graph
        # will save in the given `save_dir` or in the same directory as the npz file
        save_root = save_dir if save_dir else os.path.splitext(npz_path)[0] 
        if draw_mode in ('merge','both'):
            self.dash_print('Saving merged heatmaps...')
            # save merged heatmaps for each layer to directory with the same name as the layer
            for layer_name, heatmaps in draw_dict.items():
                save_subdir = os.path.join(save_root, 'merge', layer_name)
                os.makedirs(save_subdir, exist_ok=True)
                
                # save bgr_img only for those layers that compressed the temporal dimension, e.g. gaitgl.GLconv
                if len(heatmaps) != len(imgs):
                    compressed_imgs = temporal_align(imgs, heatmaps)
                    cimg_save_subdir = os.path.join(save_subdir, 'layer_input')
                    os.makedirs(cimg_save_subdir, exist_ok=True)
                    for frame_id, cimg in enumerate(compressed_imgs):
                        cimg_save_path = os.path.join(cimg_save_subdir, f'{frame_id}.png')
                        resized_cimg = cv2.resize(cimg, (0, 0), fx=resize_ratio, fy=resize_ratio)
                        cv2.imwrite(cimg_save_path, resized_cimg)
                else:
                    compressed_imgs = imgs

                # save each heatmap as a separate image
                for frame_id, heatmap in tqdm(enumerate(heatmaps), desc=layer_name, total=len(heatmaps)):
                    save_path = os.path.join(save_subdir, f'{frame_id}.png')
                    merge_res = (heatmap*heatmap_opacity + compressed_imgs[frame_id]*(1-heatmap_opacity)).astype('uint8') 
                    resized_frame = cv2.resize(merge_res, (0, 0), fx=resize_ratio, fy=resize_ratio)
                    cv2.imwrite(save_path, resized_frame)
            print()

        if draw_mode in ('compare','both'):
            self.dash_print('Drawing heatmap comparation graph for each frame...')
            save_subdir = os.path.join(save_root, 'compare')
            os.makedirs(save_subdir, exist_ok=True)
            
            # create a txt file indicating the mapping between layer names and layer indices
            with open(os.path.join(save_subdir, 'layer_names.txt'), 'w') as f:
                for layer_name in draw_dict.keys():
                    if layer_name == 'origin':
                        continue
                    layer_id, name = layer_name.split('_', maxsplit=1)
                    f.write(f'{layer_id}: {name}\n')
            
            # align the number of images in each layer to the minimum value, 
            # to ensure that the layer with the compressed time dimension can still be visualized.
            least_frames = min(map(lambda x: len(x), draw_dict.values()))
            if least_frames != len(imgs):
                print(
                    Panel(
                        Text('Some Operations compressed the temporal dimension, ' +\
                            f'changing the number of sequence frames from {len(imgs)} to {least_frames} !', 
                            justify="center", 
                            style='bold red italic'
                        ), 
                        title="Note",
                        box=HEAVY,
                        border_style='red'
                    )
                )

            # shrink frames to the least number & merge frames with heatmaps
            for key, value in draw_dict.items():
                if len(value) != least_frames:
                    draw_dict[key] = temporal_align(value, (least_frames, value.shape[1:])).astype('uint8')
                if key != 'origin':
                    draw_dict[key] = (draw_dict[key]*heatmap_opacity + draw_dict['origin']*(1-heatmap_opacity)).astype('uint8')
            
            draw_num = len(draw_dict)
            columns = int(np.ceil(np.sqrt(draw_num))) if draw_num >=4 else draw_num
            rows = int(np.ceil(draw_num / columns)) 
            
            for frame_id in tqdm(range(least_frames), desc='Drawing'):
                # create a canvas for each frame
                fig, axes = plt.subplots(rows, columns, figsize=(8*columns, 8*rows))
                for subplot_id in range(rows*columns): # without coordinate axes
                    axes.flat[subplot_id].axis('off')
                font_size = int(fig.get_size_inches()[0]*font_scale)
                
                # draw each layer's heatmap
                for idx, (layer_name, heatmaps) in enumerate(draw_dict.items()):
                    layer_id = layer_name.split('_',maxsplit=1)[0]
                    ax = axes.flat[idx]
                    resized_heatmap = cv2.resize(heatmaps[frame_id], (0, 0), fx=resize_ratio, fy=resize_ratio)
                    resized_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)
                    ax.imshow(resized_heatmap)
                    ax.set_title(layer_id, fontsize=font_size)

                # save comparation graph
                fig.tight_layout()
                save_path = os.path.join(save_subdir, f'{frame_id}.png')
                plt.savefig(os.path.join(save_path, save_path),dpi=300) 
                plt.close()                     

        print(f'[bold][green]Grad-CAM results saved in[/green] [magenta]{save_root}.[/]')

    def visual_tsne(self, 
                    npz_path:str,
                    embed_dim:int=2, seed:int=1597, 
                    marker_size:int=50, transparent:bool=True, title:Union[str, None]=None,
                    save_path:Union[str, None]=None, save_ext:str='png',
                    verbose:bool=True,
                    TSNE_ARGS:dict={}):
        
        """
        Perform t-SNE with given features and corresponding labels.

        Args:
            - `npz_path` (str): path of a npz file or path of the directory containing npz files. Each npz file should contain:
                - `feats` (np.ndarray): inferenced features with shape [BatchSize, FeatureDim]
                - `labels` (np.ndarray[str]): labels corresponding to the features, with shape [BatchSize] 

            - `embed_dim` (int, optional): embedding dim of t-sne, can be 2 or 3. Defaults to 2.

            - `seed` (int, optional): random seed, used to reproduce the same embedding. Defaults to 1597.

            - `marker_size` (int, optional): size of the markers in the plot. Defaults to 50.

            - `title` (Union[str, None], optional): title of the plot. Defaults to None.

            - `transparent` (bool, optional): Whether to save transparent image. Default to True.

            - `save_path` (Union[str, None], optional): path to save the result image. If it's set to None, will save the result image \
                                                        in the same directory as the input npz file. Defaults to None

            - `save_ext` (str, optional): extension of the saved image, \  
                                          support ['png','jpg','jpeg','bmp','tiff','tif','svg']. \  
                                          Defaults to 'png'.                          

            - `verbose` (bool, optional): whether to print the progress while embedding. Defaults to True.

            - `TSNE_ARGS` (dict, optional): used to receive customized TSNE drawing parameters.
        
        Returns:
            None
        """

        # recursivly draw the t-sne for all npz files in the given `npz_path` and sub-directories
        if os.path.isdir(npz_path):
            npz_paths = glob(npz_path + '/**/*.npz', recursive=True)
            if not npz_paths:
                print(f'[bold red][underline]{npz_path}[/underline] does not contain any npz file![/]')
                return
            
            console = Console()
            for npz_path in npz_paths:
                self.visual_tsne(npz_path=npz_path, 
                                 embed_dim=embed_dim, seed=seed,
                                 marker_size=marker_size, transparent=transparent,
                                 save_path=save_path, save_ext=save_ext,
                                 verbose=verbose,
                                 TSNE_ARGS=TSNE_ARGS)
                print('[green]' + '-'*console.width + '[/]')
            return

        assert embed_dim > 0, f'Target dim should be positive, but got {embed_dim}![/]'
        assert marker_size > 0, f'Marker size should be positive, but got {marker_size}![/]'
        
        if TSNE_ARGS:
            TSNE_ARGS['verbose'] = verbose
            print('[green]' + '-'*30 + '[/]')
            print('[green]Receive args for t-sne:[/]')
            for k,v in TSNE_ARGS.items():
                print(f'[green]{k}: {v}[/]')
            print('[green]' + '-'*30 + '[/]')
        
        # validate file extention 
        if not npz_path.endswith('.npz'):
            print(f'[bold red][underline]{npz_path}[/underline] is not a npz file![/]')
            return
        
        save_ext = save_ext.lower().replace('.','')
        if save_ext not in ['png','jpg','jpeg','bmp','tiff','tif','svg']:
            print(f'[bold red]Saving as `{save_ext}` file is not supported![/]')
            print('[bold dark_goldenrod]Alternativly, you can use `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `svg`[/]')
            print('[bold dark_goldenrod]Choose one and pass to `--save_ext` when using [underline]python -m visual_analyse.visualizer visual_tsne`[/]')
            return
        
        # prepare saving path
        if save_path is None:
            save_path = os.path.splitext(npz_path)[0] + '.' + save_ext
        else:
            save_path = os.path.abspath(save_path)
            save_dir = os.path.dirname(save_path)
            save_name, origin_ext = os.path.splitext(os.path.basename(save_path))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_name) + '.' + save_ext # overwrite the extention in given path with pass-in `save_ext`
            if origin_ext != save_ext:
                print(f'[bold yellow]Received `--save_ext` as {save_ext}. The saved image\'s extention is changed from {origin_ext} to {save_ext}.[/]')
            
        # load features and corresponding labels stored in npz file
        npz_path = os.path.abspath(npz_path)
        with np.load(npz_path, allow_pickle=True) as datas:
            feats = datas['feats'] # np.ndarray, (B,C,P)
            labels = datas['labels'] # np.ndarray, (B,), each element is a string     
        assert len(feats) > 1, f'There should be more than one features in the npz file, but got feature with shape {feats.shape}!'   
        
        # dimension reduction
        if TSNE_ARGS:
            tsne = TSNE(**TSNE_ARGS)
        else:
            tsne = TSNE(n_components=embed_dim, 
                        init='pca', 
                        method='exact',
                        n_jobs=-1,
                        n_iter=1000, 
                        learning_rate='auto',
                        random_state=seed,
                        verbose=verbose)

        shrink_feats = tsne.fit_transform(feats) # N, embed_dim

        # dimension check & normalize the embedding to [0,1]
        if embed_dim > 3:
            print(f'[bold yellow]⚠ Embedding dim is set to {embed_dim}, only use the [red]first 3 dims[/red] for visualization![/]')
        shrink_feats = shrink_feats[:,:3]
        x_min, x_max = np.min(shrink_feats, 0), np.max(shrink_feats, 0)
        norm_feats = (shrink_feats - x_min) / (x_max - x_min)

        # visualization
        ax = self.plot_embedding(norm_feats, labels, 
                                 marker_size=marker_size,
                                 title=title)
        
        # saving       
        ax.figure.savefig(save_path, 
                          bbox_inches='tight', 
                          dpi=300, 
                          transparent=transparent)
        
        print(f'[bold][green]T-SNE results saved in[/green] [magenta]{save_path}.\n[/]')
        print('[bold][dark_goldenrod]●  If you are not satisfied with the embedding results, ' + \
              'you can customize the embedding by passing in TSNE parameters to `-T` in this command. ' + \
              'We accept parameters of `sklearn.manifold.TSNE`, check them with this command:\n[/]')
        print('[bold gray54]    python -c "from sklearn.manifold import TSNE;print(TSNE.__doc__)"[/]')
        print('[bold gray54]    e.g. python -m visual_analyse.visualizer visual_tsne <npz_file> -T "{"perplexity":45, "random_state":1488}"\n[/]')

        if shrink_feats.shape[1] == 3:
            model_name, dataset_name = os.path.basename(npz_path)[:-4].split('_')
            ax_pkl_path = os.path.join(os.path.dirname(save_path), f'{model_name}_{dataset_name}_adjust3d.pkl')
            with open(ax_pkl_path, 'wb') as f:
                pk.dump(ax, f)
            print('[bold dark_goldenrod]●  If you are not satisfied with the 3D perspective, you can use the command below to adjust:\n[/]')
            print(f'[bold green]    python -m visual_analyse.visualizer adjust_3d [magenta]{ax_pkl_path}\n[/]')

if __name__ == '__main__':
    fire.Fire(Visualizer)