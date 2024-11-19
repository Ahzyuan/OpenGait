import os,torch,cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import autograd

def draw_CAM(layers_name:list, 
             img_path, 
             transform, 
             model, 
             save_path, max_column=5, save_soloimg=False):
    '''
    layers_name: the name list that contains all the layers you want to draw CAM for \n
    img_path: the abs path of the origin img\n
    transform: the torchvision's transform\n
    model: the model, which should generate classification probabilities\n
    save_path: the abs path of the final heatmap\n
    max_column: the max column number of the final heatmap
    '''
    def hook(module, fea_in, fea_out) -> None:
        global features_out_hook
        
        features_in_hook = fea_in       # get layer input.fea_in is a tuple object with shape of (data,)
        features_out_hook = fea_out
        return features_out_hook
  
    # init canvas
    heatmap_num = len(layers_name)
    column = min(heatmap_num+1,max_column)
    row=(heatmap_num+1)//max_column+(1 if (heatmap_num+1)%max_column else 0)
    ## Plot original image    
    fig=plt.figure(0,figsize=(10*row, 10*column))
    img_dir,img_name=os.path.split(img_path)
    image = Image.open(img_path)
    if image.mode == 'L':
        image=image.convert('RGB')
    image = transform(image)
    image = image.permute(1,2,0)    # [224,244,3] <== [3,224,224]
    ax = plt.subplot(row, column, 1)
    plt.imshow(image)
    ax.set_title(img_name,{'fontsize': 30})
    plt.axis('off')

    # draw CAM layer by layer
    model.eval()
    layer_dict = {name:layer for name, layer in model.named_modules()}
    for hm_id,layer_name in enumerate(layers_name): # 画哪一层的CAM，可通过model.named_modules()查看；若使用model.modules()，则仅显示具体各层操作，没有各层名字
        ## register hook
        assert layer_name in layer_dict,f'layer_name \'{layer_name}\' error!'
        layer_wanted = layer_dict[layer_name]
        layer_wanted.register_forward_hook(hook=hook)
        ## foreward
        output=model(image)
        features=features_out_hook
        ## get predict result
        pred = torch.argmax(output, 1).item()   #按行求最大值,即预测类别
        pred_class = output[:, pred]    #取出各样本预测类
        ## calculate grad map
        features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]   # 获取梯度
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(features_grad, (1, 1))
        pooled_grads = pooled_grads[0]  # batch size为1，去掉第0维（batch size维）
        features = features[0]
        for i in range(features.shape[0]):
            features[i,-1] *= pooled_grads[i,-1]   #逐通道梯度与图像相乘，计算各通道对识别的贡献
        ## calculate heatmap
        heatmap = features.detach().cpu().numpy()   # 将tensor格式的feature map转为numpy格式
        heatmap = np.mean(heatmap, axis=0)  # 取多通道均值
        heatmap = np.maximum(heatmap, 0)    # 取大于0的值
        heatmap /= np.max(heatmap)          # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
        
        # merge heatmap with origin img
        img = cv2.imread(img_path)  # cv2读取原图
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 调整热力图大小，与原图相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        heatmap = heatmap * 0.6+ img*0.4  # 热力图与原图融合

        # plot merged heatmap to canvas
        ax=plt.subplot(row, column, hm_id+2)
        ax.set_title(f"{layer_name}",{'fontsize': 30})
        plt.axis('off')
        fig.tight_layout()
        
        # save
        if save_soloimg:
            cv2.imwrite(os.path.join(img_dir,layer_name+'.png'), heatmap)  # 保存
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        plt.savefig(save_path,dpi=300)
    plt.close()
