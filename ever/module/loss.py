import torch
import torch.nn.functional as F
import torch.distributed as dist
from ever.core.dist import get_world_size


def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true


def all_reduce_mean(data):
    if get_world_size() == 1:
        return data
    dist.all_reduce(data, op=dist.ReduceOp.AVG)


@torch.jit.script
def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true


def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, sync_statistics=True):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0)
    if sync_statistics:
        all_reduce_mean(inter)
        all_reduce_mean(z)
    z += smooth_value

    return ((2 * inter + smooth_value) / z).mean()


def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
                          smooth_value: float = 1.0,
                          ignore_index: int = 255,
                          ignore_channel: int = -1,
                          *,
                          sync_statistics=True,
                          ):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value, sync_statistics)
    else:
        y_prob = y_pred.log_softmax(dim=1).exp()
        y_true = F.one_hot(y_true.long(), num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true.type_as(y_pred), weight, smooth_value, sync_statistics)


def tversky_loss_with_logits(
        y_pred: torch.Tensor, y_true: torch.Tensor,
        alpha: float, beta: float, gamma: float,
        smooth_value: float = 1.0,
        ignore_index: int = 255,
        *,
        sync_statistics=True,
):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    numerator = tp
    denominator = tp + alpha * fn + beta * fp

    if sync_statistics:
        all_reduce_mean(numerator)
        all_reduce_mean(denominator)

    numerator += smooth_value
    denominator += smooth_value

    tversky_coeff = numerator / denominator
    return (1. - tversky_coeff) ** gamma


@torch.jit.script
def online_hard_example_mining(losses: torch.Tensor, keep_ratio: float):
    assert 0 < keep_ratio < 1, 'The value of keep_ratio must be from 0 to 1.'
    # 1. keep num
    num_inst = losses.numel()
    num_hns = int(keep_ratio * num_inst)
    # 2. select loss
    top_loss, _ = losses.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss
    return top_loss[loss_mask].mean()


def focal_loss(y_pred, y_true, gamma: float = 2.0, normalize: bool = False):
    with torch.no_grad():
        p = y_pred.sigmoid()
        pt = (1 - p) * y_true + p * (1 - y_true)
        modulating_factor = pt.pow(gamma)

    if normalize:
        y_pred = y_pred.view(-1)
        y_true = y_true.float().view(-1)
        losses = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        modulated_losses = losses * modulating_factor
        scale = losses.sum() / modulated_losses.sum()
        return modulated_losses.sum() * scale
    else:
        return F.binary_cross_entropy_with_logits(y_pred, y_true, modulating_factor, reduction='mean')


@torch.jit.script
def sigmoid_focal_loss(
        y_pred,
        y_true,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "mean",
):
    # implementation of fvcore.nn.loss
    p = torch.sigmoid(y_pred)
    ce_loss = F.binary_cross_entropy_with_logits(
        y_pred, y_true, reduction="none"
    )
    p_t = p * y_true + (1 - p) * (1 - y_true)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def label_smoothing_cross_entropy(output: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                                  reduction: str = 'mean', ignore_index: int = -1):
    c = output.size(1)
    log_preds = F.log_softmax(output, dim=1)

    loss = -log_preds.sum(dim=1)

    loss, _ = _masked_ignore(loss, target, ignore_index)
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss * eps / c + (1 - eps) * F.nll_loss(log_preds, target, reduction=reduction, ignore_index=ignore_index)


def label_smoothing_binary_cross_entropy(output: torch.Tensor, target: torch.Tensor, eps: float = 0.1,
                                         reduction: str = 'mean', ignore_index: int = 255):
    output, target = _masked_ignore(output, target, ignore_index)
    target = torch.where(target == 0, target + eps, target - eps)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)


def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor,
                                     reduction: str = 'mean',
                                     ignore_index: int = 255,
                                     pos_weight=None
                                     ):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction, pos_weight=pos_weight)


@torch.jit.script
def soft_cross_entropy(input: torch.Tensor, target: torch.Tensor):
    assert input.dim() == 4 and target.dim() == 4
    log_probs = F.log_softmax(input, dim=1)
    return -(target * log_probs).mean(dim=(0, 2, 3)).sum()
