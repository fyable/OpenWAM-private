"""Opt-in CPU submission affinity, without moving model tensors."""

import os
import torch


def configure(device):
    groups = os.environ.get('OPENWAM_CHUNK_CPU_GROUPS', '')
    if not groups:
        return
    device = torch.device(device)
    if device.type != 'cuda':
        return
    selected = groups.split(';')[device.index or 0]
    cpus = set()
    for part in selected.split(','):
        bounds = part.split('-')
        cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    if not cpus:
        raise ValueError('Empty worker CPU affinity')
    os.sched_setaffinity(0, cpus)
