lr=1e-4
fix_layer=-1
config_file=/home/storage/diganta/mae_mtl/configs/vit/upernet_vit-b16_ln_480x480_40k_pascal_context_21.py
pretrain=/home/storage/diganta/pretrain/deit_base_patch16_384-8de9b5d1-ported.pth
data_root=/home/storage/diganta/VOCdevkit/VOC2010
work_dir=/home/storage/diganta/exps
exp_dir=pascal_context/deit_fixB_layer${fix_layer}_lr${lr}_b16_480x480

python -u tools/train.py ${config_file} \
    --work-dir ${work_dir}/${exp_dir} --seed 0 --deterministic \
    --cfg-options model.pretrained=$pretrain \
    model.backbone.fix_grad_backbone=${fix_layer} \
    data.samples_per_gpu=16 optimizer_config.cumulative_iters=8 optimizer.lr=${lr} \
    data.train.data_root=$data_root \
    data.val.data_root=$data_root \
    data.test.data_root=$data_root
