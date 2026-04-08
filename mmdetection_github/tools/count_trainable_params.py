#!/usr/bin/env python3
# Copyright (c) OpenMMLab-style utility for this project.
"""Build model from config and report trainable / total parameter counts."""
from __future__ import annotations

import argparse


def _human(n: int) -> str:
    if n >= 10**9:
        return f'{n / 10**9:.3f}B'
    if n >= 10**6:
        return f'{n / 10**6:.3f}M'
    if n >= 10**3:
        return f'{n / 10**3:.3f}K'
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Count trainable parameters for cfg.model')
    parser.add_argument('config', help='training config path')
    parser.add_argument(
        '--random-init',
        action='store_true',
        help='set backbone init_cfg to None (skip pretrained download/load)')
    parser.add_argument(
        '--by-module',
        action='store_true',
        help='break down by top-level submodules of the detector')
    args = parser.parse_args()

    from mmengine.config import Config

    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)

    if args.random_init:
        backbone = cfg.model.get('backbone')
        if isinstance(backbone, dict):
            backbone['init_cfg'] = None

    model = MODELS.build(cfg.model)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f'Config: {args.config}')
    print(f'Total parameters:      {total:>14,}  ({_human(total)})')
    print(f'Trainable parameters:  {trainable:>14,}  ({_human(trainable)})')
    print(f'Non-trainable (frozen):{total - trainable:>14,}  ({_human(total - trainable)})')

    if args.by_module:
        print('\nTop-level submodules:')
        for name, mod in model.named_children():
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in mod.parameters())
            print(f'  {name:22s}  trainable={t:>12,}  total={tot:>12,}')


if __name__ == '__main__':
    main()
