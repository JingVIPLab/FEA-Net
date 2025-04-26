import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import clip

from main import instantiate_from_config

from taming1.modules.diffusionmodules.model import Encoder, Decoder
from taming1.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming1.modules.vqvae.quantize import GumbelQuantize
from taming1.modules.vqvae.quantize import EMAVectorQuantizer


from taming1.modules.swim_network.femasr_arch import SwinLayers

from taming1.modules.vgg_network.vgg import VGGFeatureExtractor

from taming1.modules.losses_more import build_loss

cycle = ['fear', 'sadness', 'disgust', 'anger', 'amusement', 'contentment', 'awe', 'excitement']

import torch.nn as nn


def load_model(model, pretrained_dict, key):
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith(key):
            new_dict[k[len(key) + 1:]] = v
    model.load_state_dict(new_dict)

class WeightPredictor(nn.Module):
    def __init__(self,
                 in_channel,
                 cls,
                 weight_softmax=False,
                 **swin_opts,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(SwinLayers(**swin_opts))
        # weight
        self.blocks.append(nn.Conv2d(in_channel, cls, kernel_size=1))
        if weight_softmax:
            self.blocks.append(nn.Softmax(dim=1))

    def forward(self, input):
        outputs = []
        x = input
        for idx, m in enumerate(self.blocks):
            x = m(x)
        return x

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,  
                 lossconfig,  
                 emotion_class,
                 n_embed,  
                 embed_dim,  
                 ckpt_path=None,  
                 ignore_keys=[],  
                 image_key="image",  
                 file_path_="file_path_",
                 colorize_nlabels=None,  
                 monitor=None,
                 remap=None,
                 sane_index_shape=False, 
                 use_quantize=True,  
                 freeze_decoder=False,  
                 ckpt_quantize=None, 
                 ):
        super().__init__()
        self.image_key = image_key
        self.file_path_ = file_path_
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.use_quantize = use_quantize
        self.freeze_decoder = freeze_decoder
        self.emotion_class = emotion_class
        if self.use_quantize:
            self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape, device=self.device)

        # if self.freeze_decoder:
        #     checkpoint_quantize = torch.load(ckpt_quantize)['state_dict']
        #     load_model(self.quantize, checkpoint_quantize, 'quantize')

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)  
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)  
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def label_feature_extract(self, labels):
        labels = [item.split('/')[-1].split('_')[0] for item in labels]
        features_group = []
        for label in labels:  
            tokens = clip.tokenize(label) 
            text_features = self.clip_model.encode_text(tokens.to(self.device)) 
            text_features = text_features.to(torch.float32)
            text_features = text_features.mean(axis=0, keepdim=True)
            features_group.append(text_features)
        features_group_concatenated = torch.cat(features_group, dim=0) 
        labels_feature = features_group_concatenated.view(-1, 512)  
        return labels_feature

    def encode_and_decode(self, x):
        print(x.shape)
        h = self.encoder(x)  
        print(h.shape)
        bs = h.shape[0]
        h = self.quant_conv(h)
        if self.use_quantize:
            quant, emb_loss, info = self.quantize(h)  
            quant = self.post_quant_conv(quant)
            dec = self.decoder(quant)
            return dec, emb_loss, info
        else:
            return h, None, None

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec


    def forward(self, input):

        dec, diff, _ = self.encode_and_decode(input)
        return dec, diff

    
    def get_input(self, batch, k):

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_input_path(self, batch, k):
        x = batch[k]
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.decoder.conv_out.weight.requires_grad = True

        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)  
        if optimizer_idx == 0:
           
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        self.decoder.conv_out.weight.requires_grad = True

        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)  
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        parameter_list = list(self.encoder.parameters()) + \
                         list(self.quant_conv.parameters())

        if not self.freeze_decoder:
            parameter_list = parameter_list + \
                             list(self.decoder.parameters()) + \
                             list(self.post_quant_conv.parameters())
        if self.use_quantize:
            parameter_list = parameter_list + \
                             list(self.quantize.parameters())
        opt_ae = torch.optim.Adam(parameter_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)

        x = x.to(self.device)

        xrec, _ = self(x)
        if x.shape[1] > 3:

            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x




