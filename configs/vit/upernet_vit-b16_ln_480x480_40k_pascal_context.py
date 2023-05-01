_base_ = [
    '../_base_/models/upernet_vit-b16_ln.py',
    '../_base_/datasets/pascal_context.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (480, 480)

model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=crop_size,
        patch_size=16,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        out_indices=[3, 5, 7, 11],
        drop_path_rate=0.1,
        final_norm=True
    ),
    decode_head=dict(
        in_channels=[768, 768, 768, 768],
        num_classes=60,
        channels=768,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=60
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.00006,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
#     paramwise_cfg=dict(
#         custom_keys={
#             'pos_embed': dict(decay_mult=0.),
#             'cls_token': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=3e-5, 
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor', 
    paramwise_cfg=dict(
        num_layers=12, 
        layer_decay_rate=0.9
    )
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=8)

# Wandb logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook', by_epoch=False,
            init_kwargs={
                'entity': "landskape",
                'project': "mae_mtl",
                'name': "deit_fixB_layers10_lr_1e-4_b16_480x480_pascal_context",
                'config': dict(
                    model='deit_base_patch16_384',
                    dataset='pascal_context',
                    img_size=(480, 480),
                    num_fix_layers=11,
                    lr=1e-4,
                    input_resolution=(384, 384)
                )
            }
        ),
    ]
)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
