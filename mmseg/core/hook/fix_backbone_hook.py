from mmcv.runner import Hook, HOOKS


@HOOKS.register_module()
class FixBackBoneHook(Hook):
    """Hook for setting first `k` layers in backbone including 
    `patch_embed` and `pos_drop` to eval mode (freeze).

    Args:
        Hook (int): Number of layers in the backbone to freeze during training.
    """
    def __init__(self, fix_grad_backbone):
        self.fix_grad_backbone = fix_grad_backbone
    
    def before_train_iter(self, runner):
        runner.model.module.backbone.patch_embed.eval()
        runner.model.module.backbone.drop_after_pos.eval()
        for cnt, layer in enumerate(runner.model.module.backbone.layers):
            if cnt <= self.fix_grad_backbone:
                layer.eval()
