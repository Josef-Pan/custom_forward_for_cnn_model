import torch, timm
import torch.nn as nn
import torch.utils.checkpoint as cp


def vit_base_r50_s16_224(nb_classes: int, drop_dict, use_checkpoint=False) -> tuple[object, str]:
    name = 'vit_base_r50_s16_224.orig_in21k'
    # drop_rate: Head dropout rate.
    # pos_drop_rate: Position embedding dropout rate.
    # attn_drop_rate: Attention dropout rate.
    # proj_drop_rate:
    drop_rate = drop_dict.get('drop_rate', 0)
    pos_drop_rate = drop_dict.get('pos_drop_rate', 0)
    attn_drop_rate = drop_dict.get('attn_drop_rate', 0)
    proj_drop_rate = drop_dict.get('proj_drop_rate', 0)
    model = timm.create_model(name, pretrained=True, num_classes=nb_classes,
                              drop_rate=drop_rate, pos_drop_rate=pos_drop_rate,
                              attn_drop_rate=attn_drop_rate, proj_drop_rate=proj_drop_rate)
    model.saved_attn_drop = model.blocks[0].attn.attn_drop
    model.saved_proj_drop = model.blocks[0].attn.proj_drop
    if use_checkpoint:
        bound_method = forward_vit_base_r50_checkpoint.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method)
        model.check_before_block = 8  # 0...7 checked, 8...11 not checked
        modules = [model.blocks[0], model.blocks[1], model.blocks[2], model.blocks[3],
                   model.blocks[4], model.blocks[5], model.blocks[6], model.blocks[7],
                   model.blocks[8], model.blocks[9], model.blocks[10], model.blocks[11]]
        model.modules_checked = modules[:model.check_before_block]
        model.modules_not_checked = modules[model.check_before_block:]
        for i in range(model.check_before_block):  # We can't use dropout in checkpointed layers
            model.blocks[i].attn.attn_drop = nn.Dropout(0)
            model.blocks[i].attn.proj_drop = nn.Dropout(0)
    else:
        bound_method = forward_vit_base_r50_customized.__get__(model, model.__class__)
        setattr(model, 'forward', bound_method)
    return model, name


def forward_vit_base_r50_checkpoint(self, x):
    for i in range(self.check_before_block, len(self.blocks)):
        self.blocks[i].attn.attn_drop = self.saved_attn_drop if self.training else nn.Dropout(0)
        self.blocks[i].attn.proj_drop = self.saved_proj_drop if self.training else nn.Dropout(0)
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    if self.training:
        x = self.patch_drop(x)
    x = self.norm_pre(x)
    if any(prev_feature.requires_grad for prev_feature in x):
        x = cp.checkpoint_sequential(self.modules_checked, len(self.modules_checked), x)
        for module in self.modules_not_checked:
            x = module(x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    if self.global_pool:
        x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x = self.fc_norm(x)
    if self.training:
        x = self.head_drop(x)
    x = self.head(x)
    return x


def forward_vit_base_r50_customized(self, x):
    for i in range(len(self.blocks)):
        self.blocks[i].attn.attn_drop = self.saved_attn_drop if self.training else nn.Dropout(0)
        self.blocks[i].attn.proj_drop = self.saved_proj_drop if self.training else nn.Dropout(0)
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    if self.training:
        x = self.patch_drop(x)
    x = self.norm_pre(x)
    x = self.blocks(x)
    x = self.norm(x)
    if self.global_pool:
        x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x = self.fc_norm(x)
    if self.training:
        x = self.head_drop(x)
    x = self.head(x)
    return x
