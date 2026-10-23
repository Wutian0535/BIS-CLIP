"""Expression-Anchor Dynamic Masking and Identity-Decoupled OT Calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(imgs, p=16):
    """
    Convert images to patch sequences at pixel level.
    imgs: (N, 3, H, W)
    return: (N, L, patch_size**2 * 3)
    """
    h, w = imgs.shape[-2:]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = h // p
    w = w // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

class ExpressionRestoredStyleNorm(nn.Module):
    """
    Expression-Referenced Optimal Transport Calibration (ER-OTC).

    The global shallow feature distribution is treated as an identity-biased
    source distribution, while Scharr-guided expression regions define the
    expression-referenced target distribution. The implemented map is a
    diagonal Gaussian OT approximation, followed by expression-region residual
    restitution to preserve subtle FER cues.
    """

    def __init__(self, p=0.2, eps=1e-6, style_strength=0.2,
                 restitution_strength=0.7, center_sigma=0.65):
        super().__init__()
        self.p = p
        self.eps = eps
        self.ot_strength = style_strength
        self.restitution_strength = restitution_strength
        self.center_sigma = center_sigma
        self.latest_identity_style = None
        self.scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, feat, imgs):
        if not self.training:
            self.latest_identity_style = None
            return feat

        gate = self.get_expression_gate(imgs, feat.shape[-2:], feat.device, feat.dtype)
        global_mu, global_sig = self.global_style_stats(feat, detach=False)
        expr_mu, expr_sig = self.weighted_style_stats(feat, gate, detach=False)

        identity_style = torch.cat([
            (global_mu - expr_mu).flatten(1),
            (global_sig - expr_sig).flatten(1)
        ], dim=1)
        self.latest_identity_style = F.normalize(identity_style, dim=1)

        ot_global_mu, ot_global_sig = global_mu.detach(), global_sig.detach()
        ot_expr_mu, ot_expr_sig = expr_mu.detach(), expr_sig.detach()

        if torch.rand(1, device=feat.device).item() > self.p:
            return feat

        transported = self.diagonal_gaussian_ot(feat, ot_global_mu, ot_global_sig, ot_expr_mu, ot_expr_sig)
        calibrated = feat + self.ot_strength * (transported - feat)
        expr_residual = feat - calibrated
        return calibrated + self.restitution_strength * gate * expr_residual

    def global_style_stats(self, feat, detach=True):
        mu = feat.mean(dim=[2, 3], keepdim=True)
        sig = (feat.var(dim=[2, 3], keepdim=True, unbiased=False) + self.eps).sqrt()
        if detach:
            mu, sig = mu.detach(), sig.detach()
        return mu, sig

    def weighted_style_stats(self, feat, gate, detach=True):
        weight = gate / (gate.sum(dim=[2, 3], keepdim=True) + self.eps)
        mu = (feat * weight).sum(dim=[2, 3], keepdim=True)
        var = (((feat - mu) ** 2) * weight).sum(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        if detach:
            mu, sig = mu.detach(), sig.detach()
        return mu, sig

    def diagonal_gaussian_ot(self, feat, source_mu, source_sig, target_mu, target_sig):
        feat_norm = (feat - source_mu) / (source_sig + self.eps)
        return feat_norm * target_sig + target_mu

    def get_expression_gate(self, imgs, out_size, device, dtype):
        gray = imgs.mean(dim=1, keepdim=True).to(device=device, dtype=dtype)
        scharr_x = self.scharr_x.to(device=device, dtype=dtype)
        scharr_y = self.scharr_y.to(device=device, dtype=dtype)
        edge_x = F.conv2d(gray, scharr_x, padding=1)
        edge_y = F.conv2d(gray, scharr_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + self.eps)
        edge = F.interpolate(edge, size=out_size, mode='bilinear', align_corners=False)
        edge = self.min_max_norm_map(edge)

        center_prior = self.get_center_prior(out_size, device, dtype).unsqueeze(0).unsqueeze(0)
        gate = self.min_max_norm_map(edge * center_prior)
        gate = F.avg_pool2d(gate, kernel_size=3, stride=1, padding=1)
        return gate.clamp(0.0, 1.0)

    def get_center_prior(self, out_size, device, dtype):
        h, w = out_size
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, w)
        return torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.center_sigma ** 2))

    def min_max_norm_map(self, value):
        b = value.shape[0]
        value_flat = value.view(b, -1)
        value_min = value_flat.min(dim=1)[0].view(b, 1, 1, 1)
        value_max = value_flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (value - value_min) / (value_max - value_min + self.eps)

class PatchDropout_PGS(nn.Module):
    """
    Patch Generation-to-Selection module.

    Default path keeps the previous PGS behavior. When dynamic_mask is enabled,
    it keeps all tokens and applies a per-image attention mask to background
    patches selected by Scharr contour protection and manifold distance.
    """

    def __init__(self, prob, exclude_first_token=True, dynamic_mask=False,
                 dynamic_mask_min=0.05, dynamic_mask_max=0.30,
                 dynamic_mask_lambda=1.0, protect_ratio=0.30,
                 manifold_topk=8, face_threshold=0.35):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token
        self.dynamic_mask = dynamic_mask
        self.dynamic_mask_min = dynamic_mask_min
        self.dynamic_mask_max = dynamic_mask_max
        self.dynamic_mask_lambda = dynamic_mask_lambda
        self.protect_ratio = protect_ratio
        self.manifold_topk = manifold_topk
        self.face_threshold = face_threshold

        self.sinkhorn_iter = 50
        self.sinkhorn_eps = 1e-3

        self.scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]],
                                     dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, x, imgs, target_mask_ratio=0.5, min_mask_ratio=0.3,
                epoch=0, total_epochs=150, use_ed=False, use_otn=False):
        """
        x: patch embeddings (N, L+1, D), including CLS token.
        imgs: raw images (N, 3, H, W).
        """
        pixel_patches = patchify(imgs)

        if not self.training or self.prob == 0.:
            return x, None, None

        cls_tokens = x[:, :1] if self.exclude_first_token else None
        x = x[:, 1:] if self.exclude_first_token else x

        N, L, D = x.shape
        # Scharr should be computed on raw patch appearance. Normalizing each
        # patch first suppresses patch-level intensity structure and makes the
        # face-aware score unstable.
        boundary_mask = self.get_boundary_mask_from_scharr(pixel_patches) if (use_ed or self.dynamic_mask) else None
        imgs = self.normalize(pixel_patches)
        if self.dynamic_mask:
            with torch.no_grad():
                sim_matrix = self.compute_similarity(x, imgs, epoch, total_epochs)
        else:
            sim_matrix = self.compute_similarity(x, imgs, epoch, total_epochs)

        if use_otn and not self.dynamic_mask:
            sim_matrix = sim_matrix + self.sinkhorn(sim_matrix)

        if self.dynamic_mask:
            x, mask, attn_mask = self.apply_dynamic_manifold_mask(
                x, sim_matrix, L, N, boundary_mask
            )
        else:
            x, mask, attn_mask = self.apply_dropout(
                x, sim_matrix, target_mask_ratio, L, N, D, min_mask_ratio, boundary_mask
            )

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        mask_ratio = mask.sum(-1) / L
        return x, attn_mask, mask_ratio

    def normalize(self, imgs):
        mean = imgs.mean(dim=-1, keepdim=True)
        std = imgs.std(dim=-1, keepdim=True)
        return (imgs - mean) / (std + 1e-6)

    def compute_similarity(self, x, imgs, epoch, total_epochs):
        sim_x = self.cosine_similarity(x)
        sim_imgs = self.cosine_similarity(imgs)
        weight = epoch / total_epochs
        return sim_x * weight + sim_imgs * (1 - weight)

    def cosine_similarity(self, tensor):
        norm = tensor.norm(dim=-1, keepdim=True)
        tensor_normalized = tensor / (norm + 1e-6)
        return torch.bmm(tensor_normalized, tensor_normalized.transpose(1, 2))

    def generate_mask(self, x, sim_matrix, target_mask_ratio, L, N, boundary_mask):
        if boundary_mask is not None:
            boundary_mask = boundary_mask.to(x.device).view(N, L)

        len_keep = int(L * (1 - self.prob))
        noise = torch.rand(N, L, device=x.device)

        if boundary_mask is not None:
            noise = noise + boundary_mask * 1e6

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_throw = ids_shuffle[:, len_keep:]

        sim_chose = torch.gather(
            sim_matrix, dim=1, index=ids_throw.unsqueeze(-1).repeat(1, 1, L)
        )
        sim_chose_mask = (((sim_chose) > target_mask_ratio).float() * -1e12).sum(1)
        sim_chose_mask = torch.cat(
            [torch.zeros(N, 1, device=x.device, dtype=x.dtype), sim_chose_mask], dim=1
        )

        attn_mask = torch.zeros((N, L + 1, L + 1), device=x.device, dtype=x.dtype)
        attn_mask += sim_chose_mask.unsqueeze(2) + sim_chose_mask.unsqueeze(1)
        mask = (sim_chose_mask != 0)[:, 1:].type(x.dtype)
        return mask, attn_mask

    def select_mask(self, x, mask, attn_mask, L, N, D, min_mask_ratio, boundary_mask):
        noise_cutoff = torch.rand(N, L, device=x.device)
        noise_cutoff = torch.exp(noise_cutoff) * mask
        indices = torch.argsort(noise_cutoff, dim=-1)

        if boundary_mask is not None:
            min_p = L - int(mask.sum(-1).min().item()) + int(boundary_mask.sum(dim=-1).min().item())
        else:
            min_p = L - int(mask.sum(-1).min().item())

        len_keep = min(int(L * (1 - min_mask_ratio)), min_p)
        ids_keep = indices[:, :len_keep]
        ids_throw = indices[:, len_keep:]
        mask[torch.arange(N).unsqueeze(1), ids_throw] = 1

        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_attn = torch.cat(
            [torch.zeros((N, 1), device=x.device, dtype=ids_keep.dtype), ids_keep + 1], dim=1
        )
        attn_mask_enc = torch.gather(
            attn_mask, dim=1, index=ids_attn.unsqueeze(-1).repeat(1, 1, L + 1)
        )
        attn_mask_enc = torch.gather(
            attn_mask_enc, dim=2,
            index=ids_attn.unsqueeze(1).repeat(1, attn_mask_enc.shape[1], 1)
        )
        return x, mask, attn_mask_enc.detach()

    def apply_dropout(self, x, sim_matrix, target_mask_ratio, L, N, D, min_mask_ratio, boundary_mask):
        mask, attn_mask = self.generate_mask(x, sim_matrix, target_mask_ratio, L, N, boundary_mask)
        x, mask, attn_mask_enc = self.select_mask(x, mask, attn_mask, L, N, D, min_mask_ratio, boundary_mask)
        return x, mask, attn_mask_enc

    def apply_dynamic_manifold_mask(self, x, sim_matrix, L, N, boundary_mask):
        # Keep all tokens. The mask only blocks selected background patches as keys.
        boundary_mask = boundary_mask.to(x.device, dtype=x.dtype).view(N, L)
        center_prior = self.get_center_prior(L, x.device, x.dtype).unsqueeze(0)
        face_score = self.min_max_norm(boundary_mask * center_prior)

        protect_num = max(1, int(L * self.protect_ratio))
        protect_num = min(protect_num, L - 1)
        protected = torch.zeros(N, L, device=x.device, dtype=torch.bool)
        protected_ids = torch.topk(face_score, k=protect_num, dim=1).indices
        protected.scatter_(1, protected_ids, True)

        manifold_distance = self.compute_manifold_distance(sim_matrix, protected)
        mask_score = self.min_max_norm(manifold_distance + (1.0 - face_score))
        mask_score = mask_score.masked_fill(protected, -1e6)

        dynamic_ratio = self.estimate_dynamic_mask_ratio(face_score)
        max_mask_num = max(1, L - protect_num)
        mask_counts = torch.clamp((dynamic_ratio * L).long(), min=1, max=max_mask_num)

        ranked_ids = torch.argsort(mask_score, dim=1, descending=True)
        ranks = torch.arange(L, device=x.device).unsqueeze(0).expand(N, L)
        ranked_mask = ranks < mask_counts.unsqueeze(1)
        mask = torch.zeros(N, L, device=x.device, dtype=torch.bool)
        mask.scatter_(1, ranked_ids, ranked_mask)
        mask = mask & (~protected)

        attn_mask = torch.zeros(N, L + 1, L + 1, device=x.device, dtype=x.dtype)
        attn_mask[:, :, 1:] = mask.unsqueeze(1).to(x.dtype) * -1e12
        return x, mask.to(x.dtype), attn_mask

    def estimate_dynamic_mask_ratio(self, face_score):
        face_region = (face_score > self.face_threshold).to(face_score.dtype)
        background_ratio = 1.0 - face_region.mean(dim=1)
        ratio_span = self.dynamic_mask_max - self.dynamic_mask_min
        dynamic_ratio = self.dynamic_mask_min + self.dynamic_mask_lambda * ratio_span * background_ratio
        return dynamic_ratio.clamp(self.dynamic_mask_min, self.dynamic_mask_max)

    def compute_manifold_distance(self, sim_matrix, protected):
        with torch.no_grad():
            N, L, _ = sim_matrix.shape
            affinity = ((sim_matrix.detach() + 1.0) * 0.5).clamp(min=0.0)
            eye = torch.eye(L, device=sim_matrix.device, dtype=torch.bool).unsqueeze(0)
            affinity = affinity.masked_fill(eye, 0.0)

            topk = max(1, min(self.manifold_topk, L - 1))
            topk_values, topk_indices = torch.topk(affinity, k=topk, dim=-1)
            graph = torch.zeros_like(affinity)
            graph.scatter_(dim=-1, index=topk_indices, src=topk_values)
            transition = graph / (graph.sum(dim=-1, keepdim=True) + 1e-6)

            anchor_score = protected.to(sim_matrix.dtype)
            for _ in range(3):
                anchor_score = torch.bmm(transition, anchor_score.unsqueeze(-1)).squeeze(-1)
                anchor_score = torch.maximum(anchor_score, protected.to(anchor_score.dtype))

            anchor_score = self.min_max_norm(anchor_score)
            return 1.0 - anchor_score

    def get_center_prior(self, L, device, dtype):
        side_len = int(L ** 0.5)
        if side_len * side_len != L:
            raise ValueError(f"Cannot reshape {L} patches into a square grid")
        coords = torch.linspace(-1.0, 1.0, side_len, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords)
        prior = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 0.65 ** 2))
        return prior.reshape(-1)

    def min_max_norm(self, value):
        value_min = value.min(dim=1, keepdim=True)[0]
        value_max = value.max(dim=1, keepdim=True)[0]
        return (value - value_min) / (value_max - value_min + 1e-6)

    def get_boundary_mask_from_scharr(self, imgs):
        N, L, D = imgs.shape
        side_len = int(L ** 0.5)
        if side_len * side_len != L:
            raise ValueError(f"Cannot reshape {L} patches into a square image. L={L}, side_len={side_len}")

        self.scharr_x = self.scharr_x.to(imgs.dtype).to(imgs.device)
        self.scharr_y = self.scharr_y.to(imgs.dtype).to(imgs.device)

        imgs_reshaped = imgs.view(N, side_len, side_len, D)
        gray_img = imgs_reshaped.mean(dim=-1, keepdim=True)
        gray_img = gray_img.permute(0, 3, 1, 2)

        scharr_x = F.conv2d(gray_img, self.scharr_x, padding=1)
        scharr_y = F.conv2d(gray_img, self.scharr_y, padding=1)
        scharr_edges = torch.sqrt(scharr_x ** 2 + scharr_y ** 2)

        scharr_min = scharr_edges.view(N, -1).min(dim=1)[0].view(N, 1, 1, 1)
        scharr_max = scharr_edges.view(N, -1).max(dim=1)[0].view(N, 1, 1, 1)
        scharr_edges = (scharr_edges - scharr_min) / (scharr_max - scharr_min + 1e-6)
        return scharr_edges.view(N, -1)

    def sinkhorn(self, matrix):
        with torch.no_grad():
            for _ in range(self.sinkhorn_iter):
                row_sum = matrix.sum(dim=1, keepdim=True)
                col_sum = matrix.sum(dim=2, keepdim=True)

                if torch.allclose(row_sum, torch.ones_like(row_sum), atol=self.sinkhorn_eps) and \
                   torch.allclose(col_sum, torch.ones_like(col_sum), atol=self.sinkhorn_eps):
                    break

                matrix = matrix / (row_sum + self.sinkhorn_eps)
                matrix = matrix / (col_sum + self.sinkhorn_eps)
        return matrix

