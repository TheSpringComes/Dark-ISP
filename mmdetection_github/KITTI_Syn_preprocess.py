import argparse
import json
import os
from glob import glob
from multiprocessing import Pool
from os import path as osp

import cv2
import numpy as np
from scipy import stats
from tqdm import tqdm

from mmdet.datasets.transforms.noisemodel.unprocess import unprocess


def _sample_params(camera: str = 'CanonEOS5D4'):
    """Sample noise model params (matched to COCO_Syn_preprocess.py)."""
    q_step = 1
    profiles = ['Profile-1']
    saturation_level = 16383 - 512

    # Load camera calibration parameters
    base_dir = osp.dirname(__file__)
    param_dir = osp.join(
        base_dir, 'mmdet', 'datasets', 'transforms', 'noisemodel',
        'camera_params')
    camera_params = np.load(
        osp.join(param_dir, f'{camera}_params.npy'),
        allow_pickle=True).item()

    kmin = camera_params['Kmin']
    kmax = camera_params['Kmax']

    g_shape = np.random.choice(camera_params['G_shape'])
    ind = np.random.randint(0, camera_params['color_bias'].shape[0])
    color_bias = camera_params['color_bias'][ind, :]

    profile = np.random.choice(profiles)
    prof_params = camera_params[profile]

    log_k = np.random.uniform(low=np.log(kmin), high=np.log(kmax))
    log_g_scale = (
        np.random.standard_normal() * prof_params['g_scale']['sigma'] * 1
        + prof_params['g_scale']['slope'] * log_k
        + prof_params['g_scale']['bias']
    )
    log_g_scale_tl = (
        np.random.standard_normal() * prof_params['G_scale']['sigma'] * 1
        + prof_params['G_scale']['slope'] * log_k
        + prof_params['G_scale']['bias']
    )
    log_r_scale = (
        np.random.standard_normal() * prof_params['R_scale']['sigma'] * 1
        + prof_params['R_scale']['slope'] * log_k
        + prof_params['R_scale']['bias']
    )

    k = np.exp(log_k)
    g_scale = np.exp(log_g_scale)
    g_scale_tl = np.exp(log_g_scale_tl)
    r_scale = np.exp(log_r_scale)

    noise_ratio = (10, 100)
    ratio = np.random.uniform(low=noise_ratio[0], high=noise_ratio[1])

    return (k, color_bias, g_scale, g_scale_tl, g_shape, r_scale, q_step,
            saturation_level, ratio)


def _add_color_bias(img: np.ndarray, color_bias: np.ndarray) -> np.ndarray:
    c = img.shape[2]
    return img + color_bias.reshape((1, 1, c))


def _add_banding_noise(img: np.ndarray, scale: float) -> np.ndarray:
    c = img.shape[2]
    return img + np.random.randn(img.shape[0], 1, c).astype(np.float32) * scale


def add_noise(y: np.ndarray, params=None, model: str = 'PGRU'):
    """Add synthetic RAW noise (matched to COCO_Syn_preprocess.py)."""
    if params is None:
        (k, color_bias, g_scale, g_scale_tl, g_shape, r_scale, q_step,
         saturation_level, ratio) = _sample_params()
    else:
        (k, color_bias, g_scale, g_scale_tl, g_shape, r_scale, q_step,
         saturation_level, ratio) = params

    y = y * saturation_level
    y = y / ratio

    if 'P' in model:
        z = np.random.poisson(y / k).astype(np.float32) * k
    elif 'p' in model:
        z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(
            np.maximum(k * y, 1e-10))
    else:
        z = y

    if 'g' in model:
        z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(
            g_scale, 1e-10)
    elif 'G' in model:
        z = z + stats.tukeylambda.rvs(
            g_shape, loc=0, scale=g_scale_tl, size=y.shape).astype(np.float32)

    if 'B' in model:
        z = _add_color_bias(z, color_bias=color_bias)

    if 'R' in model:
        z = _add_banding_noise(z, scale=r_scale)

    if 'U' in model:
        z = z + np.random.uniform(low=-0.5 * q_step, high=0.5 * q_step)

    return z, saturation_level, ratio


def _process_one_image(img_path: str, save_folder: str, dark_ratio=(0.2, 0.5)):
    img_name, _ = osp.splitext(osp.basename(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {img_path}')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Keep behavior aligned with COCO_Syn_preprocess.py
    image = cv2.resize(image, (w * 2, h * 2))
    image = image.astype(np.float32) / 255.0

    raw, ccm, wb = unprocess(image)
    noisy_raw, wl, ratio = add_noise(raw.copy())

    _dark = np.random.uniform(dark_ratio[0], dark_ratio[1])
    noisy_raw = noisy_raw * _dark

    raw_pattern = np.array([[0, 1], [3, 2]], dtype=float)
    black_level = np.array([0, 0, 0, 0], dtype=float)
    white_level = np.array([wl, wl, wl, wl], dtype=float)
    iso = np.array([200], dtype=float)
    exp_time = np.array([0.04], dtype=float)

    os.makedirs(save_folder, exist_ok=True)
    np.savez(
        osp.join(save_folder, f'{img_name}.npz'),
        im=noisy_raw,
        ratio=ratio,
        raw_pattern=raw_pattern,
        black_level=black_level,
        white_level=white_level,
        wb=wb,
        ccm=ccm,
        iso=iso,
        exp_time=exp_time,
    )
    return img_name


def convert_kitti_coco_json_to_npz(in_json: str, out_json: str):
    """Rewrite COCO-style KITTI json: file_name *.png -> *.npz."""
    with open(in_json, 'r') as f:
        data = json.load(f)
    if 'images' not in data:
        raise KeyError(f'Invalid COCO json (missing "images"): {in_json}')

    for im in data['images']:
        fn = im.get('file_name')
        if not isinstance(fn, str):
            continue
        stem, ext = osp.splitext(fn)
        if ext.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
            im['file_name'] = f'{stem}.npz'
        elif ext.lower() == '.npz':
            pass
        else:
            im['file_name'] = f'{stem}.npz'

    os.makedirs(osp.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic RAW .npz files for KITTI image_2.')
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/jing/datasets/KITTI/',
        help='KITTI data root (should contain training/image_2/ and annotations/).'
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        default=None,
        help='Override input image dir. Default: <data_root>/training/image_2/.'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=None,
        help='Override output npz dir. Default: <data_root>/training/image_2_npz/.'
    )
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument(
        '--make-npz',
        action='store_true',
        help='Generate .npz files from images.')
    parser.add_argument(
        '--make-ann',
        action='store_true',
        help='Generate *_npz.json from existing kitti_{train,val}.json.')
    args = parser.parse_args()

    data_root = args.data_root
    img_dir = args.img_dir or osp.join(data_root, 'training', 'image_2')
    out_dir = args.out_dir or osp.join(data_root, 'training', 'image_2_npz')

    if not args.make_npz and not args.make_ann:
        # Default behavior: do both
        args.make_npz = True
        args.make_ann = True

    if args.make_ann:
        ann_dir = osp.join(data_root, 'annotations')
        for split in ('train', 'val'):
            in_json = osp.join(ann_dir, f'kitti_{split}.json')
            out_json = osp.join(ann_dir, f'kitti_{split}_npz.json')
            if osp.exists(in_json):
                convert_kitti_coco_json_to_npz(in_json, out_json)

    if args.make_npz:
        img_list = sorted(glob(osp.join(img_dir, f'*{args.suffix}')))
        pbar = tqdm(total=len(img_list), unit='image', desc='KITTI -> npz')

        def _cb(_):
            pbar.update(1)

        with Pool(args.workers) as pool:
            for p in img_list:
                pool.apply_async(
                    _process_one_image, args=(p, out_dir), callback=_cb)
            pool.close()
            pool.join()
        pbar.close()


if __name__ == '__main__':
    main()

