import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

from taming1.data.utils import custom_collate

# from taming1.distribution import save_data
# torch.manual_seed(0)
# import random
# random.seed(0)
# np.random.seed(0)

'''数据加载函数'''
# 从直方图统计生成参考张量
def ref_from_hist(model, batch_size):
    from indices import get_ref_hist
    from einops import rearrange
    hist_list = get_ref_hist()
    out = []
    for hist in hist_list:
        ref = np.argsort(-hist)[:1]
        ref = torch.from_numpy(ref).to(device).unsqueeze(1).long()
        ref = model.quantize_dec.embedding(ref) # 256*256
        ref = ref.view(1, 1, 1, -1).expand(batch_size, 16, 16, -1)
        ref = rearrange(ref, 'b h w c -> b c h w').contiguous()
        out.append(ref)
    return out

# 从前几个直方图统计生成参考张量
def ref_from_tophist(model, batch_size):
    from indices import get_ref_hist
    from einops import rearrange
    hist_list = get_ref_hist()
    out = []
    for hist in hist_list:
        ref = np.argsort(-hist)[:64]
        ref = torch.from_numpy(ref).to(device).unsqueeze(1).long()
#         ref = torch.flip(ref,dims=[0])
        ref = model.quantize_dec.embedding(ref) # 256*256*256
        ref = ref.view(1, 8, 8, -1).repeat(batch_size, 2, 2, 1)
        ref = rearrange(ref, 'b h w c -> b c h w').contiguous()
        out.append(ref)
    return out

# 将张量代码保存到磁盘。
def save_code(code, paths, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i,path in enumerate(paths):
        save_path = os.path.join(folder, os.path.basename(path)+'.npy')
        np.save(save_path, code[i, :].detach().cpu().numpy())
        print(save_path, code[i, :].detach().cpu().numpy().shape)

# 从字符串表示动态加载Python类
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

'''参数解析和配置'''
# 使用argparse模块配置参数解析器
def get_parser(**parser_kwargs):
    def str2bool(v): # 将字符串转换为布尔值
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser

'''模型和训练设置'''
# 确定非默认训练器参数
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

# 根据配置字典实例化指定的类
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

# 将对象包装为具有__len__和__getitem__的PyTorch数据集
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print("##################",self.data[idx])
        return self.data[idx]
    # def __getpath__(self, idx):
    #     # print("##################",self.data[idx]['file_path_'])
    #     return self.data[idx]['file_path_']


# 从配置创建PyTorch Lightning DataModule。
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)
            # 加载训练数据集 和 测试数据集
        # print("#######################################",data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        '''
        ssss train <taming1.data.custom.CustomTrain object at 0x7f632a58ddc0>
        ssss validation <taming1.data.custom.CustomTest object at 0x7f632a57d040>
        '''
        # print(self.datasets)
        # print(self.wrap)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
                '''
                ############ train <taming1.data.custom.CustomTrain object at 0x7f207bcfbdc0>
                ############ validation <taming1.data.custom.CustomTest object at 0x7f207bced040>
                '''

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

# 用于设置日志目录和保存配置的回调
class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)
            # print(self.logdir, self.ckptdir,self.cfgdir)
            # 设立保存的文件夹 ckpt=checkpoints cfg=configs

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

# 在训练期间记录图像的回调
class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, train=True, batch_size=1):
        super().__init__()
        self.batch_freq = batch_frequency
        self.batch_size = batch_size
        if not train:
            self.batch_freq = 1000000
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        # print("save_dir",save_dir,"split",split,"images", images)
        # ,save_dir：保存的文件夹,split：train还是test, images全部图片的tensor
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.batch_size, padding=0)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "monitor": "val/aeloss_epoch",
#                 "monitor": "val/loss",
                "save_top_k": 3,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 256,
                    "clamp": True,
                    "train": opt.train,
                    "batch_size": config.data.params.batch_size,
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        
        # data
        data = instantiate_from_config(config.data)
        # main.DataModuleFromConfig object at 0x7fb7eb589f40

        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train: # 训练部分
            try:
                # print("adsadsadsadsadasdsa")
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted: # 测试部分
            # print("ccccccccccccccccccccccccc")
            device = torch.device('cuda:1')
            model = model.to(device)
            model.eval()
            loader = DataLoader(data.datasets["test"], batch_size=data.batch_size,collate_fn=custom_collate)
            
#             ref_list = ref_from_tophist(model, data.batch_size)
            
            for batch_idx, batch in enumerate(loader):
                print('Testing', batch_idx, '/', len(loader))
                
                ### Log images ###
                with torch.no_grad():
#                     log = model.log_images_with_ref(batch, ref=ref_list)
                    log = model.log_images(batch)
                root = os.path.join(logdir, "images", 'test')
                os.makedirs(root, exist_ok=True)
                
                for k in log.keys():
                    image = None
                    grid = torchvision.utils.make_grid(log[k], nrow=data.batch_size, padding=0)
                    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                    grid = torch.clip(grid, 0, 1)
                    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                    grid = grid.detach().cpu().numpy()
                    grid = (grid*255).astype(np.uint8)
                    if image is None:
                        image = grid
                    else:
                        image = np.concatenate((image, grid), 0)
                    filename = k+"_{:06}.png".format(batch_idx)

#                     filename = os.path.basename(batch['image1_path'][0])+"_"+os.path.basename(batch['image2_path'][0])+".png"
                    
                    path = os.path.join(root, filename)
                    Image.fromarray(image).save(path)
                    
                del log
                if batch_idx>1000:
                    break
                    
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

# save_data()