# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from sklearn.decomposition import PCA
import torch
from einops import rearrange
import numpy as np


def _th_pca(feature: torch.Tensor, k, **kwargs) -> torch.Tensor:
    b, c, h, w = feature.shape
    A = rearrange(feature, 'b c h w -> b (h w) c')
    U, S, V = torch.pca_lowrank(A, q=kwargs.get('q', None), niter=kwargs.get('niter', 2))

    # explained_variance = S ** 2 / (A.size(1) - 1)
    # total_var = explained_variance.sum()
    # explained_variance_ratio = explained_variance / total_var
    # print(explained_variance_ratio[:, :k])
    pc = A @ V[:, :, :k]
    pc = rearrange(pc, 'b (h w) c -> b c h w', h=h, w=w)
    return pc


def _sk_pca(feature: torch.Tensor, k) -> torch.Tensor:
    b, c, h, w = feature.shape
    A = rearrange(feature, 'b c h w -> b (h w) c').cpu().numpy()

    pc = np.zeros([b, k, h, w], dtype=A.dtype)
    for i, A_i in enumerate(A):
        pca_fn = PCA(n_components=k)
        Ak_i = pca_fn.fit_transform(A_i)
        Ak_i = rearrange(Ak_i, '(h w) c -> c h w', h=h, w=w)
        pc[i] = Ak_i
    return torch.from_numpy(pc)


def pca(feature: torch.Tensor, k, backend='sk', normalize=True, to_255=False):
    assert feature.ndim == 4

    if backend == 'sk':
        pc = _sk_pca(feature, k)
    elif backend == 'th':
        pc = _th_pca(feature, k)
    else:
        raise NotImplementedError

    if normalize:
        h, w = pc.shape[-2:]
        pc = rearrange(pc, 'b c h w -> b c (h w)')
        pc -= pc.min(dim=2, keepdim=True).values
        pc /= pc.max(dim=2, keepdim=True).values
        pc = rearrange(pc, 'b c (h w) -> b c h w', h=h, w=w)

    if to_255:
        pc = pc.mul_(255).clip_(0, 255).to(torch.uint8)
    return pc
