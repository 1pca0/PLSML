import numpy as np
import torch
import torch.nn.functional as F


smooth = 1.
epsilon = 1e-6

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def l2_normalize(x, dim=None, eps=1e-12):
    """Normalize a tensor over dim using the L2-norm."""
    sq_sum = torch.sum(torch.square(x), dim=dim, keepdim=True)
    inv_norm = torch.rsqrt(torch.max(sq_sum, torch.ones_like(sq_sum)*eps))
    return x * inv_norm


def all_gather(tensor, expand_dim=0, num_replicas=None):
    """Gathers a tensor from other replicas, concat on expand_dim and return."""
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    return torch.cat([o.unsqueeze(expand_dim) for o in other_replica_tensors], expand_dim)


class NT_Xent(nn.Module):
    """Wrap a module to get self.training member."""

    def __init__(self):
        super(NT_Xent, self).__init__()

    def forward(self, embedding1, embedding2, temperature=0.2, num_replicas=1):
        """NT-XENT Loss from SimCLR
        :param embedding1: embedding of augmentation1
        :param embedding2: embedding of augmentation2
        :param temperature: nce normalization temp
        :param num_replicas: number of compute devices
        :returns: scalar loss
        :rtype: float32
        """
        batch_size = embedding1.shape[0]
        feature_size = embedding1.shape[-1]
        num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        LARGE_NUM = 1e9

        # normalize both embeddings
        embedding1 = l2_normalize(embedding1, dim=-1)
        embedding2 = l2_normalize(embedding2, dim=-1)

        if num_replicas > 1 and self.training:
            # First grab the tensor from all other embeddings
            embedding1_full = all_gather(embedding1, num_replicas=num_replicas)
            embedding2_full = all_gather(embedding2, num_replicas=num_replicas)

            # fold the tensor in to create [B, F]
            embedding1_full = embedding1_full.reshape(-1, feature_size)
            embedding2_full = embedding2_full.reshape(-1, feature_size)

            # Create pseudo-labels using the current replica id & ont-hotting
            replica_id = dist.get_rank()
            labels = torch.arange(batch_size, device=embedding1.device) + replica_id * batch_size
            labels = labels.type(torch.int64)
            full_batch_size = embedding1_full.shape[0]
            masks = F.one_hot(labels, full_batch_size).to(embedding1_full.device)
            labels = F.one_hot(labels, full_batch_size * 2).to(embedding1_full.device)
        else:  # no replicas or we are in test mode; test set is same size on all replicas for now
            embedding1_full = embedding1
            embedding2_full = embedding2
            masks = F.one_hot(torch.arange(batch_size), batch_size).to(embedding1.device)
            labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(embedding1.device)

        # Matmul-to-mask
        logits_aa = torch.matmul(embedding1, embedding1_full.T) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(embedding2, embedding2_full.T) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(embedding1, embedding2_full.T) / temperature
        logits_ba = torch.matmul(embedding2, embedding1_full.T) / temperature

        # Use our standard cross-entropy loss which uses log-softmax internally.
        # Concat on the feature dimension to provide all features for standard softmax-xent
        loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss = loss_a + loss_b
        return torch.mean(loss)

def sad_loss(pred, target, encoder_flag=True):
	"""
	AD: atention distillation loss
	: param pred: input prediction
	: param target: input target
	: param encoder_flag: boolean, True=encoder-side AD, False=decoder-side AD
	"""
	target = target.detach()
	if (target.size(-1) == pred.size(-1)) and (target.size(-2) == pred.size(-2)):
		# target and prediction have the same spatial resolution
		pass
	else:
		if encoder_flag == True:
			# target is smaller than prediction
			# use consecutive layers with scale factor = 2
			target = F.interpolate(target, scale_factor=2, mode='trilinear')
		else:
			# prediction is smaller than target
			# use consecutive layers with scale factor = 2
			pred = F.interpolate(pred, scale_factor=2, mode='trilinear')

	num_batch = pred.size(0)
	pred = pred.view(num_batch, -1)
	target = target.view(num_batch, -1)
	pred = F.softmax(pred, dim=1)
	target = F.softmax(target, dim=1)
	return F.mse_loss(pred, target)

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss(pred, target):
	"""
	DSC loss
	: param pred: input prediction
	: param target: input target
	"""
	iflat = pred.view(-1)
	tflat = target.view(-1)
	intersection = torch.sum((iflat * tflat))
	return 1. - ((2. * intersection + smooth)/(torch.sum(iflat) + torch.sum(tflat) + smooth))


def binary_cross_entropy(y_pred, y_true):
	"""
	Binary cross entropy loss
	: param y_pred: input prediction
	: param y_true: input target
	"""
	y_true = y_true.view(-1).float()
	y_pred = y_pred.view(-1).float()
	return F.binary_cross_entropy(y_pred, y_true)


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
	"""
	Focal loss
	: param y_pred: input prediction
	: param y_true: input target
	: param alpha: balancing positive and negative samples, default=0.25
	: param gamma: penalizing wrong predictions, default=2
	"""
	# alpha balance weight for unbalanced positive and negative samples
	# clip to prevent NaN's and Inf's
	y_pred_flatten = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
	y_pred_flatten = y_pred_flatten.view(-1).float()
	y_true_flatten = y_true.detach()
	y_true_flatten = y_true_flatten.view(-1).float()
	loss = 0

	idcs = (y_true_flatten > 0)
	y_true_pos = y_true_flatten[idcs]
	y_pred_pos = y_pred_flatten[idcs]
	y_true_neg = y_true_flatten[~idcs]
	y_pred_neg = y_pred_flatten[~idcs]

	if y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0:
		# positive samples
		logpt = torch.log(y_pred_pos)
		loss += -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha

	if y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0:
		# negative samples
		logpt2 = torch.log(1. - y_pred_neg)
		loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)

	return loss



def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w x d
        output: batch_size x 1 x h x w x d
    """
    assert v.dim() == 5
    n, c, h, w, d = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * d * np.log2(c))



def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3],f_.shape[4]) + 1e-8

def similarity(feat):
    feat = feat.float()
    # tmp = L2(feat).detach()
    tmp = L2(feat)
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2]*f_T.shape[-3])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis
def CriterionPairWiseforWholeFeatAfterPool(preds_S, preds_T):
    feat_S = F.softmax(preds_S, dim=1)
    feat_T = F.softmax(preds_T, dim=1)

    loss = 0
    # maxpool = nn.AvgPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=0, ceil_mode=True) # change
    ix_all = np.random.choice(7, 2, replace=False)
    iy_all = np.random.choice(7, 2, replace=False)
    iz_all = np.random.choice(5, 1, replace=False)
    #        ix_all = [1]
    #        iy_all = [1]
    #        iz_all = [1]
    for ix in ix_all:
        for iy in iy_all:
            for iz in iz_all:
                sub_feat_S = feat_S[:, :, (ix * 16):((ix + 1) * 16), (iy * 16):((iy + 1) * 16),
                             (iz * 16):((iz + 1) * 16)]
                sub_feat_T = feat_T[:, :, (ix * 16):((ix + 1) * 16), (iy * 16):((iy + 1) * 16),
                             (iz * 16):((iz + 1) * 16)]

                sub_loss = sim_dis_compute(sub_feat_S, sub_feat_T)
                loss = loss + sub_loss
    return loss / (2 * 2)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w, d = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def selfinformation(preds_S, preds_T):
        feat_S = prob_2_entropy(F.softmax(preds_S, dim=1))
        feat_T = prob_2_entropy(F.softmax(preds_T, dim=1))

        loss = torch.mean((feat_S - feat_T) ** 2) / 2

        return loss

def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent
# def focal_loss(y_pred, y_true, alpha=0.75, gamma=2.0):
# 	"""
# 	Focal loss
# 	: param y_pred: input prediction
# 	: param y_true: input target
# 	: param alpha: balancing positive and negative samples, default=0.75
# 	: param gamma: penalizing wrong predictions, default=2
# 	"""
# 	# alpha balance weight for unbalanced positive and negative samples
# 	# clip to prevent NaN's and Inf's
# 	y_pred = F.relu(y_pred)
# 	y_pred = torch.clamp(y_pred, min=epsilon, max=1.-epsilon)
# 	y_true = y_true.view(-1).float()
# 	y_pred = y_pred.view(-1).float()
# 	idcs = (y_true > 0)
# 	y_true_pos = y_true[idcs]
# 	y_pred_pos = y_pred[idcs]
# 	y_true_neg = y_true[~idcs]
# 	y_pred_neg = y_pred[~idcs]
#
# 	if (y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0) and (y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0):
# 		# positive samples
# 		logpt = torch.log(y_pred_pos)
# 		loss = -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha
# 		# negative samples
# 		logpt2 = torch.log(1. - y_pred_neg)
# 		loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)
# 		return loss
# 	else:
# 		# use binary cross entropy to avoid NaN/Inf caused by missing positive or negative samples
# 		return F.binary_cross_entropy(y_pred, y_true)