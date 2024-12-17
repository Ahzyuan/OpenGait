# Author: Ahzyuan
# Date: 2024.12.02
# coding: utf-8

import os,pickle,json,math,pdb
import torch
import torch.nn as nn
import torch.utils.data as tordata
from tqdm import tqdm
from rich import print
from opengait.data.collate_fn import CollateFn
from opengait.modeling import models as gaitmodels
from opengait.data.transform import get_transform
from opengait.modeling.base_model import BaseModel
from opengait.utils import get_valid_args

class SimpleModel(BaseModel, nn.Module):
    
    def __init__(self, args):
        super(BaseModel, self).__init__() # init nn.Module
        
        cfgs = args.cfgs
        concrete_model = getattr(gaitmodels, cfgs['model_cfg']['model'])
        method_lists = [method_name for method_name,method_handle in concrete_model.__dict__.items() if not method_name.startswith('_')]
        for method in method_lists: # inherit from concrete gait model
            setattr(self, method, self.capture_features(getattr(concrete_model, method)))
        
        self.cfgs = cfgs    
        self.engine_cfg = cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        self.save_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        self.no_inplace()

        self.samples_loader = self.get_loader(
            cfgs['data_cfg'], train=False)
        self.evaluator_trfs = get_transform(
            cfgs['evaluator_cfg']['transform'])

        self.device = args.device # torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device("cuda", self.device))

        self.train(False) # self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)
    
    def capture_features(self, func):
        def wrapper(*args, **kwargs):
            return func(self, *args, **kwargs)
        return wrapper        
    
    def no_inplace(self):
        toggle = 0
        for name, module in self.named_modules():
            if not module._modules and hasattr(module, 'inplace'):
                module.inplace = False
                toggle = 1
        if toggle: 
            print("[bold green]Turn off inplace operation in all modules[/]")
    
    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = NoMsgDataSet(data_cfg, train)
        self.dataset = dataset

        vaild_args = get_valid_args(InferenceSampler, sampler_cfg, free_keys=[
            'sample_type', 'type'])
        sampler = InferenceSampler(dataset, **vaild_args) # Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg['num_workers'])
        return loader
    
    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']

        checkpoint = torch.load(save_name, map_location=torch.device(
            "cuda", self.device))
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            print("-------- Restored Params List --------")
            print(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                print(
                    "Restore NO Scheduler from %s !!!" % save_name)
        print("Restore Parameters from %s !!!" % save_name)
    
    def inference_step(self, ipts, pick_key=None):
        retval = self.forward(ipts)
        inference_feat = retval['inference_feat'] 
        img = retval['visual_summary']['image/sils'] # (T,1,H,W), torch.float32
        del retval
        
        # for k, v in inference_feat.items():
        #     inference_feat[k] = ts2np(v)
        
        if pick_key is None:
            if len(inference_feat) == 1:
                pick_key = tuple(inference_feat.keys())[0]
            else:
                key_idx = input('\nChoose the key of real inference feature:\n' + \
                    '\n'.join([f'{idx+1}: {key}' for idx, key in enumerate(inference_feat.keys())]) +\
                    '\nEnter the number: ')
                pick_key = tuple(inference_feat.keys())[int(key_idx) - 1]
            
        final_features = inference_feat[pick_key] # batch_size, out_channels, parts_num
        
        return final_features, img, pick_key
    
    def inference(self):
        total_size = len(self.samples_loader)
        pbar = tqdm(total=total_size, desc='Infering')

        batch_size = self.samples_loader.batch_sampler.batch_size
        rest_size = total_size
        pick_key = None
    
        for inputs in self.samples_loader: # inputs(tuple): batch_data:list, batch_labels:list[str], batch_types:list[str], batch_views:list[str], sequence_length:ndarray(1,1)
            ipts = self.inputs_pretreament(inputs) # ipts: transformed_data:list, labels:list[str], types:list[str], views:list[str], sequence_length:ndarray(1,1)

            final_features, img, pick_key = self.inference_step(ipts, pick_key)
            
            #yield final_features, img
            
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        
class NoMsgDataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"] + partition["TEST_SET"]
        test_set = partition["TRAIN_SET"] + partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in test_set]

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                print('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                print(pid_list)

        if len(miss_pids) > 0:
            print('-------- Miss Pid List --------')
            print(miss_pids)
        if training:
            print("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            print("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(os.path.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(os.path.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = os.path.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [os.path.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            print('Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)

class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = 1 # dist.get_world_size() 
        rank = 0# dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)