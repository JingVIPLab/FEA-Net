from copy import deepcopy

from taming1.modules.utils import get_root_logger
from taming1.modules.utils.registry import LOSS_REGISTRY
from taming1.modules.losses_more.losses import (CharbonnierLoss, GANLoss, ContrastiveLoss, L1Loss, MSELoss, SoftCrossEntropy, PerceptualLoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)

__all__ = [
    'ContrastiveLoss', 'L1Loss', 'MSELoss', 'SoftCrossEntropy', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
