# Author: Ahzyuan
# Date: 2024.12.02
# coding: utf-8

import os,cv2,fire,pdb
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from rich import print
from typing import Union
from rich.box import HEAVY
from rich.text import Text
from rich.panel import Panel
from rich.console import Console

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
'''

class Visualizer:
    def dash_print(self,string:str):
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
            npz_path (str): path of a npz file or path of the directory containing npz files.
            
            save_dir (Union[str,None], optional): path of a directory to save result images. When it's set to None,
                will be save to the same directory as the npz file with the npz file's name.Defaults to None.
                
            heatmap_opacity (float, optional): opacity of the heatmap. Defaults to 0.6.
            
            draw_mode (str, optional): the mode to draw heatmap, can be 'merge', 'compare' or 'both'. When it's set to 'merge',
                will only save each heatmap as a separate image in the subdirectory with corresponding layer name. When it's set
                to 'compare', will draw origin image as well as all different layers' heatmaps in a figure. When it's set to 'both',
                will perform the behavior that 'merge' and 'compare' will do, respectively. Defaults to 'merge'.
                
            resize_ratio (float, optional): image resize ratio. Defaults to 1.0.
            
            font_scale (float, optional): font size scale ratio. This parameter only takes effect on 'compare' mode. Defaults to 2.5.
        
        Returns:
            None
        """
        assert heatmap_opacity >= 0 and heatmap_opacity <= 1, f'heatmap_opacity should be in range [0,1], but got {heatmap_opacity}!'
        assert draw_mode in ['merge', 'compare', 'both'], f'draw_mode should be in [merge, compare, both], but got {draw_mode}!'
        
        # recursivly draw the heatmap for all npz files in the given `npz_path` and sub-directories
        if os.path.isdir(npz_path):
            npz_paths = glob(npz_path + '/**/*.npz',recursive=True)
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

if __name__ == '__main__':
    fire.Fire(Visualizer)