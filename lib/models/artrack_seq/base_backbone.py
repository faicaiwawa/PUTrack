from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.models.artrack_seq.prompter import Prompter
from lib.models.layers.patch_embed import PatchEmbed
from lib.models.artrack_seq.utils import combine_tokens, recover_tokens

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

        depth = 12
        block_nums = depth
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        prompt_mixer_blocks = []
        for i in range(block_nums):
            prompt_mixer_blocks.append(
                Prompter(dim=768))
        self.prompt_mixer_blocks = nn.Sequential(*prompt_mixer_blocks)

        prompt_norms = []
        for i in range(block_nums):
            prompt_norms.append(norm_layer(768))
        self.prompt_norms = nn.Sequential(*prompt_norms)

        prompt_feature_gate = []
        for i in range(block_nums):
            prompt_feature_gate.append(Gate_Feature(NUM=256, GATE_INIT=10, NUM_TOKENS=256))
        self.prompt_feature_gate = nn.Sequential(*prompt_feature_gate)

        prompt_gates_x = []
        for i in range(block_nums - 1):
            prompt_gates_x.append(Gate_Prompt(GATE_INIT=10, NUM_TOKENS=256))
        self.prompt_gates_x = nn.Sequential(*prompt_gates_x)

        #self.prompt_tokens = nn.Parameter(torch.zeros(12, 5, 768))

        #import math
        #from functools import reduce
        #from operator import mul
        #val = math.sqrt(6. / float(3 * reduce(mul, (16, 16), 1) + 768))  # noqa
        #nn.init.uniform_(self.prompt_tokens, -val, val)

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    def forward_features(self, z, x, identity):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        x_ = self.prompt_norms[0](x)

        x_ = self.prompt_mixer_blocks[0](token2feature(x_))

        x_ = feature2token(x_)

        x = self.prompt_feature_gate[0](x, x_)  # 前面主题 后面附加

        z += self.pos_embed_z
        x += self.pos_embed_x

        z += identity[:, 0, :].repeat(B, self.pos_embed_z.shape[1], 1)
        x += identity[:, 1, :].repeat(B, self.pos_embed_x.shape[1], 1)

        #prompt_token_num = self.prompt_tokens.shape[1]
        #prompt_token = self.prompt_tokens[0].unsqueeze(0).expand(x.shape[0], -1, -1)

        x = combine_tokens(z, x, mode=self.cat_mode)
        lens_z = self.pos_embed_z.shape[1]
        #x = torch.cat((x, prompt_token), dim=1)
        x = self.pos_drop(x)


        for i, blk in enumerate(self.blocks):
            if (i >= 1):
                z = x[:, :lens_z]
                x_ori = x[:, lens_z:]
                x = self.prompt_norms[i](x_ori)

                x_weight_cur = feature2token(self.prompt_mixer_blocks[i](token2feature(x)))

                x_ = self.prompt_gates_x[i - 1](x_, x_weight_cur)

                x = self.prompt_feature_gate[i](x_ori, x_)

                x = combine_tokens(z, x, mode=self.cat_mode)

                #prompt_token = self.prompt_tokens[i].unsqueeze(0).expand(x.shape[0], -1, -1)
                #x = torch.cat((x, resv_prompt, prompt_token), dim=1)
            x = blk(x)
            #if i == 0:  # 320 + 5
            #    resv_prompt = x[:, 320:]  # prompt[0]
            #    x = x[:, :320]  # 320
            #else:  # 320 + 5 + 5
            #    resv_prompt = x[:, 320 + prompt_token_num:]  # keep right side 5 tokens alive
            #    x = x[:, : 320]
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        # x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict

    def forward(self, z, x, identity, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, identity)

        return x, aux_dict

class Gate_Feature(nn.Module):
    def __init__(self, NUM=320, GATE_INIT=10, NUM_TOKENS=320):
        super().__init__()
        self.num = NUM
        gate_logit = (torch.ones(NUM) * GATE_INIT)
        self.num_tokens = NUM_TOKENS
        self.gate_logit = nn.Parameter(gate_logit)

    def forward(self, xin, xout):
        gate = self.gate_logit.sigmoid()
        gate = gate.unsqueeze(0).unsqueeze(-1).repeat(xin.size(0), 1, xin.size(2))
        prompt_out = gate * xout
        prompt_in = 1 * xin
        xout = prompt_out + prompt_in
        return xout


class Gate_Prompt(nn.Module):
    def __init__(self, NUM=1, GATE_INIT=10, NUM_TOKENS=256):
        super().__init__()
        gate_logit = -(torch.ones(NUM) * GATE_INIT)
        self.num_tokens = NUM_TOKENS
        self.gate_logit = nn.Parameter(gate_logit)

    def forward(self, xin, xout):
        gate = self.gate_logit.sigmoid()
        prompt_in = xin
        prompt_out = xout
        xout = (1 - gate) * prompt_out + gate * prompt_in
        return xout
def token2feature(tokens):
    B, L, D = tokens.shape
    H = W = int(L ** 0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x

def feature2token(x):
    B, C, W, H = x.shape
    L = W * H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens