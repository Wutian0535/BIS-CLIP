from collections import OrderedDict
import torch
import torch.nn as nn
from models.expression_anchor_masking import PatchDropout_PGS, ExpressionRestoredStyleNorm


class CLIPVIT(nn.Module):
    def __init__(self, args, clip_model, embed_dim=768,):
        super().__init__()

        self.final_dim = 512
        # self.final_dim = 768  # for vit-l/14

        # visual
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.use_clip_proj = True  # 默认用CLIP的proj

        # 获取transformer的层数
        self.num_layers = len(self.transformer.resblocks)

        if not self.use_clip_proj:  # 如果不用CLIP的proj, 则用自己的proj
            self.projection = nn.Sequential(OrderedDict([  # No
                ('fc1', nn.Linear(embed_dim, self.final_dim)),
                ('act', nn.Tanh()),
                ('fc2', nn.Linear(self.final_dim, self.final_dim)),
            ], ))

        self.projection_dist = clip_model.visual.proj  # 全局头用1层fc


        # text
        self.clip_text_encoder = clip_model.encode_text

        # 存储预训练模型的引用，用于蒸馏
        self.teacher_clip_model = clip_model

        # ===== Expression-restored style normalization =====
        self.use_expr_restitution = getattr(args, 'use_expr_restitution', False)
        if self.use_expr_restitution:
            self.expr_restitution = ExpressionRestoredStyleNorm(
                p=getattr(args, 'expr_restitution_p', 0.2),
                style_strength=getattr(args, 'expr_style_strength', 0.2),
                restitution_strength=getattr(args, 'expr_restitution_strength', 0.7),
                center_sigma=getattr(args, 'expr_restitution_center_sigma', 0.65),
            )
            print(f"[ER-OTC] Enabled with p={getattr(args, 'expr_restitution_p', 0.2)}, "
                  f"ot_strength={getattr(args, 'expr_style_strength', 0.2)}, "
                  f"residual_strength={getattr(args, 'expr_restitution_strength', 0.7)}, "
                  f"center_sigma={getattr(args, 'expr_restitution_center_sigma', 0.65)}")
        # ===== PGS (Patch Generation-to-Selection) 配置 =====
        self.use_pgs = getattr(args, 'use_pgs', False)
        if self.use_pgs:
            patch_dropout_prob = getattr(args, 'force_patch_dropout', 0.05)
            self.use_dynamic_mask = getattr(args, 'use_dynamic_mask', False)
            self.patch_dropout_pgs = PatchDropout_PGS(
                patch_dropout_prob,
                dynamic_mask=self.use_dynamic_mask,
                dynamic_mask_min=getattr(args, 'dynamic_mask_min', 0.05),
                dynamic_mask_max=getattr(args, 'dynamic_mask_max', 0.30),
                dynamic_mask_lambda=getattr(args, 'dynamic_mask_lambda', 1.0),
                protect_ratio=getattr(args, 'protect_ratio', 0.30),
                manifold_topk=getattr(args, 'manifold_topk', 8),
                face_threshold=getattr(args, 'face_threshold', 0.35),
            )
            self.target_mask_ratio = getattr(args, 'target_mask_ratio', 0.3)
            self.min_mask_ratio = getattr(args, 'min_mask_ratio', 0.1)
            self.use_ed = getattr(args, 'use_ed', False)
            self.use_otn = getattr(args, 'use_otn', False)
            self.total_epochs = getattr(args, 'epochs', 150)
            # 获取注意力头数，用于 attn_mask 的 repeat
            # ViT-B/16: width=768, head_width=64, 所以 heads=12
            self.num_heads = clip_model.visual.conv1.out_channels // 64
            print(f"[PGS] Enabled with target_mask_ratio={self.target_mask_ratio}, "
                  f"min_mask_ratio={self.min_mask_ratio}, use_ed={self.use_ed}, "
                  f"use_otn={self.use_otn}, dynamic_mask={self.use_dynamic_mask}")
        # ===================================================

    def forward_features(self, x, layer_idx=-1, epoch=0, raw_imgs=None):
        """
        提取特征，支持从指定层提取中间层特征
        Args:
            x: 输入图像 (bs, 3, H, W)
            layer_idx: 指定提取哪一层的特征，-1表示最后一层，0表示第一层
            epoch: 当前训练 epoch（PGS 需要）
            raw_imgs: 原始图像 (bs, 3, H, W)（PGS 需要，用于边缘检测和像素相似度）
        Returns:
            x: 提取的特征 (bs, num_tokens, embed_dim)
        """
        # ===== PGS: 保存原始图像用于边缘检测和像素相似度 =====
        attn_mask = None
        raw_input = x

        x = self.conv1(x)  # shape = [*, width, grid, grid], (bs,768,14,14)
        if self.use_expr_restitution:
            x = self.expr_restitution(x, raw_imgs if raw_imgs is not None else raw_input)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2], (bs,768,196)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width], (bs,196,768)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width], (bs,197,768)
        x = x + self.positional_embedding.to(x.dtype)

        # ===== PGS: 在位置编码之后、Transformer 之前执行掩码 =====
        if self.use_pgs and self.training and raw_imgs is not None:
            x, attn_mask, mask_ratio = self.patch_dropout_pgs(
                x, raw_imgs,
                target_mask_ratio=self.target_mask_ratio,
                min_mask_ratio=self.min_mask_ratio,
                epoch=epoch,
                total_epochs=self.total_epochs,
                use_ed=self.use_ed,
                use_otn=self.use_otn
            )
            # 将 attn_mask 转换为 MultiheadAttention 需要的格式
            if attn_mask is not None:
                # attn_mask: (N, L+1, L+1) -> (N*num_heads, L+1, L+1).
                # Keep the additive -inf-style mask instead of converting it
                # to bool; masked patches are suppressed as attention keys.
                attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
                attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        # ==========================================================

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, (L+1, bs, 768)

        # 支持从中间层提取特征，传入动态注意力掩码
        stop_idx = (self.num_layers + layer_idx) % self.num_layers
        for i, resblock in enumerate(self.transformer.resblocks):
            x = resblock(x, dynamic_attn_mask=attn_mask)
            if i == stop_idx:
                break

        x = x.permute(1, 0, 2)  # LND -> NLD  (bs, L+1, 768)

        x = self.ln_post(x)
        return x

    def forward_text(self, label_token):
        with torch.no_grad():
            x = self.clip_text_encoder(label_token)

        return x

    def get_identity_style(self):
        if not getattr(self, 'use_expr_restitution', False):
            return None
        return getattr(self.expr_restitution, 'latest_identity_style', None)

    def forward(self, x, label_token, norm_pred=True, layer_idx=-1, epoch=0):
        """
        前向传播，支持从中间层提取特征
        Args:
            x: 输入图像
            label_token: 标签token
            norm_pred: 是否对预测结果归一化
            layer_idx: 指定提取哪一层的特征，-1表示最后一层
            epoch: 当前训练 epoch（PGS 需要）
        """
        # text forward
        label_embed = self.forward_text(label_token)
        label_embed = label_embed / label_embed.norm(dim=-1, keepdim=True)

        # image forward - 支持从中间层提取特征，传入 epoch 和原始图像用于 PGS
        raw_imgs = x if self.use_pgs else None
        x = self.forward_features(x, layer_idx=layer_idx, epoch=epoch, raw_imgs=raw_imgs)
        dist_feat = x[:, 0] @ self.projection_dist  # (bs,512), x的第一列特征(class token) 做线性变换(图中e_cls)
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
        # dist_feat = self.dist_norm(dist_feat)  # 二选一即可,都用效果不好88-89

        # Global Head Only
        score = dist_feat @ label_embed.t()
        if norm_pred:
            score = score / score.norm(dim=-1, keepdim=True)

        return score, None, dist_feat

    # 仅image encoder, test时用
    def encode_img(self, x, layer_idx=-1):
        """
        编码图像，支持从中间层提取特征
        Args:
            x: 输入图像
            layer_idx: 指定提取哪一层的特征，-1表示最后一层
        注意：推理时不使用 PGS，走完整路径
        """
        x = self.forward_features(x, layer_idx=layer_idx)
        dist_feat = x[:, 0] @ self.projection_dist
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
        return None, dist_feat



    def extract_teacher_features(self, x, layer_idx=-1):
        """
        从预训练CLIP模型中提取中间层特征，用于蒸馏
        Args:
            x: 输入图像 (bs, 3, H, W)
            layer_idx: 要提取的层索引，-1表示最后一层
        Returns:
            teacher_feat: 预训练模型的特征 (bs, num_tokens, embed_dim)
        """
        with torch.no_grad():
            # 使用预训练模型的前处理
            x = self.teacher_clip_model.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.teacher_clip_model.visual.class_embedding.to(x.dtype) + 
                         torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                         x], dim=1)
            x = x + self.teacher_clip_model.visual.positional_embedding.to(x.dtype)
            x = self.teacher_clip_model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)

            # 处理到指定层
            stop_idx = (self.num_layers + layer_idx) % self.num_layers
            for i, resblock in enumerate(self.teacher_clip_model.visual.transformer.resblocks):
                x = resblock(x)
                if i == stop_idx:
                    break

            x = x.permute(1, 0, 2)
            x = self.teacher_clip_model.visual.ln_post(x)

            # 提取全局特征并投影
            teacher_feat = x[:, 0, :] @ self.teacher_clip_model.visual.proj
            teacher_feat = teacher_feat / teacher_feat.norm(dim=-1, keepdim=True)

        return teacher_feat





