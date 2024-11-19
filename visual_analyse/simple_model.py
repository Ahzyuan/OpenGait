import argparse, os, torch
from tqdm import tqdm
from opengait.utils import config_loader
from opengait.data.transform import get_transform
from opengait.modeling.base_model import BaseModel

class SimpleModel(BaseModel):
    
    def __init__(self, args):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(SimpleModel, self).__init__()
        cfgs = args.cfgs
        self.cfgs = cfgs
        self.engine_cfg = cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        self.save_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])

        self.build_network(cfgs['model_cfg'])
        self.init_parameters()

        if self.engine_cfg['with_test']:
            self.samples_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])

        self.device = cfgs.device # torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device("cuda", self.device))

        self.train(False) # self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)
    
    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.samples_loader)
        pbar = tqdm(total=total_size, desc='Transforming')
        # if rank == 0:
        #     pbar = tqdm(total=total_size, desc='Transforming')
        # else:
        #     pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help='Path of the config file.')
    parser.add_argument('-s', '--sample_dir', type=str, default='visual_analyse/sample_data',
                        help='Path of the sample data which will be analysed. Note that the structure of the dir should be the same as a dataset.')
    parser.add_argument('--device',  type=int, default=0,
                        help='Device index to use.')
    args = parser.parse_args()
    args.config_path = os.path.abspath(args.config_path)
    args.sample_dir = os.path.abspath(args.sample_dir)
    assert os.path.exists(args.config_path), f"{args.config_path} does not exist."
    assert os.path.exists(args.sample_dir), f"{args.sample_dir} not found!"
    
    cfgs = config_loader(args.config_path)
    cfgs['data_cfg']['dataset_root'] = args.sample_dir
    args.cfg = cfgs
    
    model = SimpleModel(args)