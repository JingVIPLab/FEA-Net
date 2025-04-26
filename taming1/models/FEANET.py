import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from main import instantiate_from_config

from taming1.modules.diffusionmodules.model import Encoder, Decoder, FDAM
from taming1.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming1.modules.swim_network.femasr_arch import SwinLayers
import clip

import numpy as np


from taming1.distribution import data_embed_collect, label_collect

from taming1.modules.losses.weight_loss import SoftCrossEntropy as WeightCE

cycle = ['fear', 'sadness', 'disgust', 'anger', 'joy', 'surprise']


def disable_grad(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


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


class FEANET(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 emotion_class,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key1="image1",
                 image_key2="image2",
                 image_path1="image1_path",
                 image_path2="image2_path",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=True,  # tell vector quantizer to return indices as bhw
                 checkpoint_source=None,
                 checkpoint_ref=None,
                 use_residual=True,
                 use_conv=True,
                 use_selfatt=True,
                 ):
        super().__init__()
        self.image_key1 = image_key1
        self.image_key2 = image_key2
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self.encoder = Encoder(**ddconfig)
        self.quant_conv_source = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.encoder_real = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.emotion_class = emotion_class
        self.e_dim = embed_dim

        self.in_ch = 256
        self.cls = emotion_class
        self.weight_softmax = True
        self.weight_predictor = WeightPredictor(
            self.in_ch,
            self.cls,
            self.weight_softmax
        )
        self.weight_predictor_fuse = WeightPredictor(
            self.in_ch,
            self.cls,
            self.weight_softmax
        )

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap,
                                         sane_index_shape=sane_index_shape,device=self.device)

        quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap,
                                         sane_index_shape=sane_index_shape,device=self.device)
        before_quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

        self.quant_conv = nn.ModuleList()
        self.quantize_group = nn.ModuleList()
        for i in range(self.emotion_class):
            self.quantize_group.append(quantize)
            self.quant_conv.append(before_quant_conv)

        self.fdam = FDAM(embed_dim, block_num=10, residual=use_residual, use_conv=use_conv,
                                             use_selfatt=use_selfatt)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss = instantiate_from_config(lossconfig)

        if checkpoint_source is not None and ckpt_path is None:
            print('loading stage1 checkpoint from', checkpoint_source)
            ckpt_source = torch.load(checkpoint_source, map_location='cpu')['state_dict']
            load_model(self.encoder, ckpt_source, 'encoder')
            load_model(self.quant_conv_source, ckpt_source, 'quant_conv')

            load_model(self.post_quant_conv, ckpt_source, 'post_quant_conv')
            load_model(self.decoder, ckpt_source, 'decoder')

            try:
                load_model(self.quantize, ckpt_source, 'quantize')
                print("successful loaded")
            except:
                print("******fail*******")
                pass

        if checkpoint_ref is not None and ckpt_path is None:
            print('loading stage2 checkpoint from', checkpoint_ref)
            ckpt_ref = torch.load(checkpoint_ref, map_location='cpu')['state_dict']
            load_model(self.encoder_real, ckpt_ref, 'encoder')
            load_model(self.quant_conv, ckpt_ref, 'quant_conv')
            load_model(self.weight_predictor, ckpt_ref, 'weight_predictor')
            print('loaded 6 codebook checkpoint from', checkpoint_ref)
            try:
                load_model(self.quantize_group, ckpt_ref, 'quantize_group')
                print("successful loaded codebook", checkpoint_ref)
            except:
                print("******fail*******")
                pass
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

            self.set_module_param_freeze(self.quantize_group, False)
            self.set_module_param_freeze(self.quant_conv, True)
            self.set_module_param_freeze(self.post_quant_conv, True)
            self.set_module_param_freeze(self.encoder, True)
            self.set_module_param_freeze(self.weight_predictor, False)
            self.set_module_param_freeze(self.weight_predictor_fuse, True)
            self.set_module_param_freeze(self.decoder, True)
            self.set_module_param_freeze(self.fdam, True)

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor


        self.softCE = WeightCE(loss_weight=0.1)
    def set_module_param_freeze(self, module, freeze=False):
        for param in module.parameters():
            param.requires_grad = freeze  



    def get_tensor(self, label):
        vector = [0.02 if x != label else 0.9 for x in cycle]
        tensor = torch.tensor(vector, dtype=torch.float32)
        return tensor

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

    def encode(self, x, quantize=True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        if not quantize:
            return h, 0, [0, 0, 0]
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    @torch.no_grad()
    def encode_real(self, x, quantize=True):
        h = self.encoder_real(x)
        quant_group = []


        feat_before_codebook = []
        for idx in range(self.emotion_class):
            feat = self.quant_conv[idx](h)
            feat_before_codebook.append(feat)

        for idx in range(self.emotion_class):
            quant, emb_loss, info = self.quantize_group[idx](feat_before_codebook[idx])
            quant_group.append(quant)



        return h, quant_group, (None, None, None)

     

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def encode_to_z(self, x):
        quant_z, _, info = self.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    def forward(self, input):
        quant, diff, _ = self.encode(input)  
        dec = self.decode(quant)  
        return dec, diff

    def AFAM(self, ref_weight, out):

        weight = self.weight_predictor_fuse(out).unsqueeze(2)  
        weight_loss = self.softCE(weight.squeeze(2), ref_weight.squeeze(2))
        return weight_loss, weight

    def transfer(self, x, ref, quantize=True):
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            # source
            h = self.encoder(x)
            h1 = self.quant_conv_source(h)
            quant_x, _, _ = self.quantize(h1)
            quant_x = quant_x.detach()
            h_ref, quant_group, info_ref = self.encode_real(ref, quantize=True)  
            ref_weight = self.weight_predictor(h_ref).unsqueeze(2) 
            x = torch.sum(torch.mul(torch.stack(quant_group).transpose(0, 1), ref_weight), dim=1)
            quant_ref = x
            indices_ref = info_ref[2]
            quant_ref = quant_ref.detach()

        h_x = self.fdam(quant_x, quant_ref)


        diff_x2y_loss = []
        quant_group = []

        for idx in range(self.emotion_class):
            quant, emb_loss, info = self.quantize_group[idx](h_x, None)
            quant_group.append(quant)
            diff_x2y_loss.append(emb_loss)

        diff_x2y = sum(diff_x2y_loss) / 6


        weight_loss, weight = self.AFAM(ref_weight, h_x)

        x = torch.sum(torch.mul(torch.stack(quant_group).transpose(0, 1), weight), dim=1)
        quant_y = x

        indices_y = (None, None, None,)
        if weight_loss > 0:
            diff_x2y = diff_x2y + weight_loss
        else:
            diff_x2y = diff_x2y

        return quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref

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
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)


        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2)


        if optimizer_idx == 0:
            # autoencode
            total_loss, aeloss, log_dict_ae = self.loss(diff_x2y,
                                                        quant_ref, quant_y,
                                                        quant_x,
                                                        indices_ref, indices_y,
                                                        optimizer_idx, self.global_step,
                                                        last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return total_loss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(diff_x2y,
                                                quant_ref, quant_y,
                                                quant_x,
                                                indices_ref, indices_y,
                                                optimizer_idx, self.global_step, last_layer=self.get_last_layer(),
                                                split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)


        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2)

        total_loss, aeloss, log_dict_ae = self.loss(diff_x2y,
                                                    quant_ref, quant_y,
                                                    quant_x,
                                                    indices_ref, indices_y,
                                                    0, self.global_step, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(diff_x2y,
                                            quant_ref, quant_y,
                                            quant_x,
                                            indices_ref, indices_y,
                                            1, self.global_step, last_layer=self.get_last_layer(), split="val")
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate

        parameter_list = list(self.fdam.parameters()) \
                + list(self.encoder.parameters()) \
                + list(self.post_quant_conv.parameters()) \
                + list(self.decoder.parameters()) \
                + list(self.weight_predictor_fuse.parameters()) \
                + list(self.quant_conv.parameters())



        opt_ae = torch.optim.Adam(parameter_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.quantize.embedding.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x1 = self.get_input(batch, self.image_key1)
        x2 = self.get_input(batch, self.image_key2)
        labels1 = self.get_input_path(batch, self.image_path1)  # path1
        labels2 = self.get_input_path(batch, self.image_path2)  # path2
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        # train
        label1 = [item.split('/')[-1].split('_')[0] for item in labels1]
        label2 = [item.split('/')[-1].split('_')[0] for item in labels2]

        # test
        # label1 = [item.split('/')[-1] for item in labels1]
        # label2 = [item.split('/')[-1] for item in labels2]
        # label1 = label1[0]
        # label2 = label2[0]

        label1 = str(label1)
        label2 = str(label2)
        label = label1 + "-" + label2

        quant_x, quant_y, quant_ref, diff_x2y, indices_y, indices_ref = self.transfer(x1, x2)  # EDITED
        x2_out = self.decode(quant_y)
        log[label] = torch.cat((x1, x2, x2_out))
        # log[label] = x2_out
        return log