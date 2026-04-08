_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

kitti_classes = (
    'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
    'Misc')

model = dict(
    type='RAWRetinaNet',
    data_preprocessor=dict(
        type='DetRAWDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='Dark_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        model='our',
        ISP_version='v2',
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=len(kitti_classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(type='PseudoSampler'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=10),
    cri_pix=dict(type='L1Loss', loss_weight=1.0))

dataset_type = 'CocoDataset'
data_root = '/home/jing/datasets/KITTI/'

backend_args = None

train_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args, postfix='npz'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'gt', 'iso', 'wb',
                   'white_level', 'black_level', 'ccm', 'raw_pattern',
                   'exp_time', 'ratio'))
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args, postfix='npz'),
    dict(type='Resize', scale=(1242, 375), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'iso', 'wb', 'white_level', 'ratio',
                   'black_level', 'ccm', 'raw_pattern', 'exp_time'),
        bayer_substitute='/home/jing/datasets/KITTI/training/image_2/',
        suffix='.png')
]

metainfo = dict(classes=kitti_classes)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/kitti_train_npz.json',
        data_prefix=dict(img='training/image_2_npz/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/kitti_val_npz.json',
        data_prefix=dict(img='training/image_2_npz/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/kitti_val_npz.json', metric='bbox')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 20
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
]
train_cfg = dict(max_epochs=max_epochs)

auto_scale_lr = dict(base_batch_size=16)

find_unused_parameters = True

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
)
