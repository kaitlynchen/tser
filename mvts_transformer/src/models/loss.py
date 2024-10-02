import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss_module(config):

    task = config['task']

    if (task == "imputation") or (task == "transduction"):
        return MaskedMSELoss(reduction='none')  # outputs loss for each batch element

    if task == "classification":
        return NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample

    if task == "regression":
        return nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        # return nn.SmoothL1Loss(reduction='none')
    else:
        raise ValueError("Loss module for task '{}' does not exist".format(task))


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


def l1_reg_loss(model):
    """
    Returns L1 norm of some linear layers
    """
    l1_norm = 0.
    n_params = 0.
    for name, param in model.named_parameters():
        # HACK list some names for now
        # In Torch MultiHeadAttention, linear layers are called self_attn.in_proj..., self_attn.out_proj
        # In TransformerBatchNormEncoderLayer, linear layers are called linaer1/linear2
        # In Attention_Rel_Scl, they are under query/key/value
        if "linear1" in name or "linear2" in name or "self_attn" in name or "query" in name or "key" in name or "value" in name or "conv" in name:
            l1_norm += param.abs().sum()
            n_params += param.numel()
    return l1_norm / n_params


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
