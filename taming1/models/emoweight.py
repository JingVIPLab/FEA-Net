import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import clip

from main import instantiate_from_config

from taming1.modules.diffusionmodules.model import Encoder, Decoder
from taming1.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming1.modules.vqvae.quantize import GumbelQuantize
from taming1.modules.vqvae.quantize import EMAVectorQuantizer
from taming1.modules.vqvae.template import imagenet_templates

from taming1.modules.swim_network.femasr_arch import SwinLayers

from taming1.modules.vgg_network.vgg import VGGFeatureExtractor

from taming1.modules.utils.registry import ARCH_REGISTRY


cycle = ['fear', 'sadness', 'disgust', 'anger', 'joy', 'surprise']

import torch.nn as nn


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def load_model(model, pretrained_dict, key):
    model_dict = model.state_dict()
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith(key):
            new_dict[k[len(key)+1:]] = v
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


class WeightPredictor1(nn.Module):
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
                 freeze_decoder=False, 
                 ckpt_quantize=None, 
                 checkpoint=True,
                 codebook_ckpt=[],
                 ):
        super().__init__()
        self.image_key = image_key
        self.file_path_ = file_path_
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.freeze_decoder = freeze_decoder
        self.emotion_class = emotion_class

        self.in_ch = 256
        self.cls = emotion_class
        self.weight_softmax = True
        self.weight_predictor = WeightPredictor(
            self.in_ch,
            self.cls,
            self.weight_softmax
        )

        self.quantize_group = nn.ModuleList()
        self.quant_conv = nn.ModuleList()

        quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap,
                                   sane_index_shape=sane_index_shape, device=self.device)
        before_quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

        # define module
        for i in range(self.emotion_class):
            self.quantize_group.append(quantize)
            self.quant_conv.append(before_quant_conv)

        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # ckpt_path
        print('loaded 1 encoder and 6 quan_conv and 1 post_quant_conv and 1 decoder checkpoint from', ckpt_path)
        ckpt_gan = torch.load(ckpt_path, map_location='cpu')['state_dict']
        load_model(self.encoder, ckpt_gan, 'encoder')
        load_model(self.post_quant_conv, ckpt_gan, 'post_quant_conv')
        load_model(self.decoder, ckpt_gan, 'decoder')
        for idx in range(self.emotion_class):
            load_model(self.quant_conv[idx], ckpt_gan, 'quant_conv')

        print('loaded 6 codebook checkpoint from', codebook_ckpt)
        for idx in range(self.emotion_class):
            cbckpt = torch.load(codebook_ckpt[idx], map_location='cpu')['state_dict']
            try:
                load_model(self.quantize_group[idx], cbckpt, 'quantize')
                print("successful loaded", codebook_ckpt[idx])
            except:
                print("******fail*******")
                pass

        self.set_module_param_freeze(self.quantize_group, False)
        self.set_module_param_freeze(self.quant_conv, True)
        self.set_module_param_freeze(self.post_quant_conv, True)
        self.set_module_param_freeze(self.encoder, True)
        self.set_module_param_freeze(self.weight_predictor, True)
        self.set_module_param_freeze(self.decoder, True)


        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor


        use_semantic_loss = True
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 256, 1, 1, 0),
                nn.ReLU(),
                )
            self.vgg_feat_layer = 'relu5_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])
            self.set_module_param_freeze(self.vgg_feat_extractor, False)

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


    def set_module_param_freeze(self, module, freeze=False):
        for param in module.parameters():
            param.requires_grad = freeze  
        

    def get_tensor(self, label):
        vector = [0.02 if x != label else 0.9 for x in cycle]
        tensor = torch.tensor(vector, dtype=torch.float32)
        return tensor

    def feature_loss(self, z, z_gt):
        """
        Args:
            z: lq features BxCxHxW
            z_gt: gt features BxCxHxW
        """
        z = z.permute(0, 2, 3, 1).contiguous()
        z_gt = z_gt.permute(0, 2, 3, 1).contiguous()
        feature_loss = 0.25 * ((z_gt.detach() - z) ** 2).mean()
        feature_loss += self.gram_loss(z, z_gt.detach())
        return feature_loss

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()


    def encode_and_decode(self, x):
        h = self.encoder(x) 
        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(x)[self.vgg_feat_layer]
                vgg_feat = self.conv_semantic(vgg_feat)
 
        emb_loss_b = []
        quant_group = []

        feat_before_codebook = []
        for idx in range(self.emotion_class):
            feat = self.quant_conv[idx](h)
            feat_before_codebook.append(feat)

        
        for idx in range(self.emotion_class):
            quant, emb_loss, info = self.quantize_group[idx](feat_before_codebook[idx])
            quant_group.append(quant)
            emb_loss_b.append(emb_loss)




        weight = self.weight_predictor(h).unsqueeze(2)  
        x = torch.sum(torch.mul(torch.stack(quant_group).transpose(0, 1), weight), dim=1)
        quant = x

        quant = self.post_quant_conv(quant)

        emb_loss = torch.sum(torch.stack(emb_loss_b, dim=0), dim=0)


        dec = self.decoder(quant)

        highfeat_loss = F.mse_loss(quant, vgg_feat) * 0.05

        return dec, emb_loss + highfeat_loss, info


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
        # x1 = self.get_input_path(batch, self.file_path_) # label

        xrec, qloss = self(x) # 重构输出 (xrec) 和量化损失 (qloss)
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
        parameter_list = list(self.encoder.parameters())

        parameter_list = parameter_list + list(self.quant_conv.parameters())

#         if not self.freeze_decoder:
        parameter_list = parameter_list + \
                         list(self.decoder.parameters()) + \
                         list(self.post_quant_conv.parameters())
#       if self.use_quantize:
        parameter_list = parameter_list + list(self.weight_predictor.parameters())


        # opt_ae = torch.optim.Adam(parameter_list,
        #                           lr=lr, betas=(0.5, 0.9))
        # opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
        #                             lr=lr, betas=(0.5, 0.9))

        opt_ae = torch.optim.Adam(parameter_list,
                                  lr=lr, betas=(0.9, 0.99))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.9, 0.99))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        # x1 = self.get_input_path(batch, self.file_path_)  # label
        x = x.to(self.device)
        # x1= x1.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
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
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x




