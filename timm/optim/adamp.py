"""
AdamP Optimizer Implementation copied from https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

Paper: `Slowing Down the Weight Norm Increase in Momentum-based Optimizers` - https://arxiv.org/abs/2006.08217
Code: https://github.com/clovaai/AdamP

Copyright (c) 2020-present NAVER Corp.
MIT license

Modified by YANG Ruixin for gradient centralization
2021/03/18
https://github.com/yang-ruixin
yang_ruixin@126.com (in China)
rxn.yang@gmail.com (out of China)
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
import math

from .centralization import centralized_gradient  # ================================


class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False,
                 use_gc=False, gc_conv_only=False, gc_loc=False):  # ================================
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

        # ================================
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        self.gc_loc = gc_loc
        # ================================

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = x.norm(dim=1).add_(eps)
        y_norm = y.norm(dim=1).add_(eps)
        dot = (x * y).sum(dim=1)

        return dot.abs() / x_norm / y_norm

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # ================================
                # GC operation
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                # ================================

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # ================================
                # GC operation
                # perturb aka G_grad
                if self.gc_loc is False:
                    perturb = centralized_gradient(perturb, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                # ================================

                # Step
                p.data.add_(-step_size, perturb)

        return loss
