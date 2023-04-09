lr=1e-4
fix_layer=6
config_file=/data/jaygala/mae_mtl_exps/mmsegmentation/configs/vit/upernet_vit-b16_ln_640x640_160k_ade20k.py
pretrain=/data/jaygala/mae_mtl_exps/pretrain/dino_vit_base_patch16_224.pth
data_root=/data/jaygala/mae_mtl_exps/ADEChallengeData2016
work_dir=/data/jaygala/mae_mtl_exps/exps
exp_dir=dino_fixB_layer${fix_layer}_lr${lr}_b16_640x640_ade20k


nohup python -u tools/train.py ${config_file} \
    --work-dir ${work_dir}/${exp_dir} --seed 0 --deterministic \
    --cfg-options model.pretrained=$pretrain \
    model.backbone.fix_grad_backbone=${fix_layer} \
    data.samples_per_gpu=2 optimizer.lr=${lr} \
    data.train.data_root=$data_root \
    data.val.data_root=$data_root \
    data.test.data_root=$data_root > logs/$exp_dir.log &
