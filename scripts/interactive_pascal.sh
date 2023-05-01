lr=1e-4
fix_layer=5
config_file=/data/jaygala/mae_mtl_exps/mmsegmentation/configs/vit/upernet_vit-b16_ln_480x480_40k_pascal.py
pretrain=/data/jaygala/mae_mtl_exps/pretrain/deit_base_patch16_384-8de9b5d1-ported.pth
data_root=/data/jaygala/mae_mtl_exps/pascal_context/VOCdevkit/VOC2010
work_dir=/data/jaygala/mae_mtl_exps/exps
exp_dir=pascal_context/deit_384_fixB_layer${fix_layer}_lr${lr}_b16_480x480

CUDA_VISIBLE_DEVICES=0 python -u tools/train.py ${config_file} \
    --work-dir ${work_dir}/${exp_dir} --seed 0 --deterministic \
    --cfg-options model.pretrained=$pretrain \
    model.backbone.fix_grad_backbone=${fix_layer} \
    data.samples_per_gpu=8 optimizer_config.cumulative_iters=8 optimizer.lr=${lr} \
    data.train.data_root=$data_root \
    data.val.data_root=$data_root \
    data.test.data_root=$data_root
