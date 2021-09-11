""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman

Modified by YANG Ruixin for outputting learning rate, SimCLR loss and classification loss
2021/09/07
https://github.com/yang-ruixin
yang_ruixin@126.com (in China)
rxn.yang@gmail.com (out of China)
"""
import csv
import os
from collections import OrderedDict


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, eval_metrics, lr, simclr_loss, classification_loss, filename, write_header=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])

    # ================================
    rowd.update(OrderedDict(lr=lr))
    rowd.update(OrderedDict(simclr_loss=simclr_loss))
    rowd.update(OrderedDict(classification_loss=classification_loss))
    # ================================

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
