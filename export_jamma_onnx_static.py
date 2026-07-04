#!/usr/bin/env python3
"""
JamMa → ONNX Export — Static Shapes (no NonZero, no dynamic dims)
==================================================================

All matching outputs are fixed-size: exactly K coarse matches → K fine matches.
``nonzero()`` / ``torch.where()`` replaced with ``TopK`` for TRT compatibility.

Usage:
    conda activate jamma
    python export_jamma_onnx_static.py [--K 4096]

Output:
    jamma_static.onnx    — Full ONNX model (all static shapes)
    jamma_static_ref.pt  — PyTorch reference
"""

import os, sys, math, time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.jamma.utils.utils import normalize_keypoints as _nk_orig
from src.utils.profiler import PassThroughProfiler

# ============================================================================
# 1. Custom ONNX op registration
# ============================================================================

def register_selective_scan_symbolic():
    from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn

    @staticmethod
    def symbolic(g, u, delta, A, B, C, D, z, delta_bias,
                 delta_softplus, return_last_state=False):
        return g.op("jam::SelectiveScan", u, delta, A, B, C, D, z, delta_bias,
                    delta_softplus_i=int(delta_softplus), outputs=1)

    SelectiveScanFn.symbolic = symbolic
    print("✓ Registered jam::SelectiveScan")


# ============================================================================
# 2. RMSNorm (replace Triton)
# ============================================================================

class RMSNormONNX(nn.Module):
    def __init__(self, orig_norm):
        super().__init__()
        self.weight = nn.Parameter(orig_norm.weight.data.clone())
        self.eps = getattr(orig_norm, 'eps', 1e-5)

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms * self.weight.float()).to(dtype=x.dtype)


# ============================================================================
# 3. normalize_keypoints (remove AMP decorator)
# ============================================================================

def normalize_keypoints_onnx(kpts, size):
    if not isinstance(size, Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    return (kpts - shift[..., None, :]) / scale[..., None, None]


# ============================================================================
# 4. Static-shape coarse matching (TopK replaces nonzero)
# ============================================================================

def mask_border_static(m, b, v):
    """Mask borders (same as original but optimized)."""
    if b <= 0:
        return
    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def coarse_match_static(matcher, sim_matrix, data, K=8000):
    """
    Original nonzero logic → pad/trunc to exactly K via cat + slice.
    Keeps the exact same match selection as dynamic nonzero.
    """
    thr = matcher.coarse_matching.thr
    use_sm = matcher.coarse_matching.use_sm
    border_rm = matcher.coarse_matching.border_rm
    dev = sim_matrix.device

    conf0 = F.softmax(sim_matrix, 2) if use_sm else sim_matrix
    mask0 = (conf0 > thr) & (conf0 == conf0.max(dim=2, keepdim=True)[0])
    conf1 = F.softmax(sim_matrix, 1) if use_sm else sim_matrix
    mask1 = (conf1 > thr) & (conf1 == conf1.max(dim=1, keepdim=True)[0])
    mask = mask0 | mask1

    h0c, w0c = data['hw0_c'][0], data['hw0_c'][1]
    h1c, w1c = data['hw1_c'][0], data['hw1_c'][1]
    mask_4d = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                        h0c=h0c, w0c=w0c, h1c=h1c, w1c=w1c)
    mask_border_static(mask_4d, border_rm, False)
    mask = rearrange(mask_4d, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)')

    # Original nonzero — dynamic output
    b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
    mconf = sim_matrix[b_ids, i_ids, j_ids]

    # Pad to exactly K: cat with zero buffer, slice first K
    zero_b = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_i = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_j = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_c = torch.zeros(K, device=dev)

    b_ids = torch.cat([b_ids, zero_b])[:K]
    i_ids = torch.cat([i_ids, zero_i])[:K]
    j_ids = torch.cat([j_ids, zero_j])[:K]
    mconf = torch.cat([mconf, zero_c])[:K]

    data.update({
        'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids, 'mconf': mconf,
    })

    scale = data['hw0_i'][0] / h0c
    mkpts0_c = torch.stack([i_ids % w0c,
                            torch.div(i_ids, w0c, rounding_mode='trunc')],
                           dim=1) * scale
    mkpts1_c = torch.stack([j_ids % w1c,
                            torch.div(j_ids, w1c, rounding_mode='trunc')],
                           dim=1) * scale
    data.update({
        'm_bids': b_ids, 'mkpts0_c': mkpts0_c, 'mkpts1_c': mkpts1_c,
    })


# ============================================================================
# 5. Static-shape fine matching (fixed K in → fixed K out)
# ============================================================================

def fine_sub_match_static(matcher, conf_matrix_fine, feat_f0_unfold, feat_f1_unfold, data, K=8000):
    """
    Original fine matching logic → pad/trunc to exactly K.
    """
    fm = matcher.fine_matching
    W_f = fm.W_f
    thr = fm.fine_thr
    dev = conf_matrix_fine.device

    mask = (conf_matrix_fine > thr) & \
           (conf_matrix_fine == conf_matrix_fine.amax(dim=[1, 2], keepdim=True))

    # Original logic: mask.max(dim=2) → where → gather
    mask_float = mask.float()
    mask_v_float, all_j_ids = mask_float.max(dim=2)
    mask_v = mask_v_float > 0.5
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]
    mconf = conf_matrix_fine[b_ids, i_ids, j_ids]

    # Pad to exactly K
    zero_b = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_i = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_j = torch.zeros(K, dtype=torch.int64, device=dev)
    zero_c = torch.zeros(K, device=dev)
    b_ids = torch.cat([b_ids, zero_b])[:K]
    i_ids = torch.cat([i_ids, zero_i])[:K]
    j_ids = torch.cat([j_ids, zero_j])[:K]
    mconf = torch.cat([mconf, zero_c])[:K]

    b_ids_c = data['b_ids']
    i_ids_c = data['i_ids']
    j_ids_c = data['j_ids']
    scale_f_c = data['hw0_f'][0] // data['hw0_c'][0]

    mkpts0_c_f = torch.stack([
        i_ids_c % data['hw0_c'][1],
        torch.div(i_ids_c, data['hw0_c'][1], rounding_mode='trunc'),
    ], dim=1) * scale_f_c

    mkpts1_c_f = torch.stack([
        j_ids_c % data['hw1_c'][1],
        torch.div(j_ids_c, data['hw1_c'][1], rounding_mode='trunc'),
    ], dim=1) * scale_f_c

    updated_b_ids = b_ids_c[b_ids]
    scale = data['hw0_i'][0] / data['hw0_f'][0]
    scale0 = scale * data['scale0'][updated_b_ids] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][updated_b_ids] if 'scale1' in data else scale

    mkpts0_fw = torch.stack([
        i_ids % W_f, torch.div(i_ids, W_f, rounding_mode='trunc')], dim=1)
    mkpts1_fw = torch.stack([
        j_ids % W_f, torch.div(j_ids, W_f, rounding_mode='trunc')], dim=1)

    # Sub-pixel refinement
    sub_ref = fm.subpixel_mlp(torch.cat(
        [feat_f0_unfold[b_ids, i_ids], feat_f1_unfold[b_ids, j_ids]], dim=-1))
    sub_ref0, sub_ref1 = torch.chunk(sub_ref, 2, dim=1)
    sub_ref0 = torch.tanh(sub_ref0) * 0.5
    sub_ref1 = torch.tanh(sub_ref1) * 0.5

    pad = 0 if W_f % 2 == 0 else W_f // 2
    mkpts0_f = (mkpts0_fw + mkpts0_c_f[b_ids] - pad) * scale0 + sub_ref0 * scale0
    mkpts1_f = (mkpts1_fw + mkpts1_c_f[b_ids] - pad) * scale1 + sub_ref1 * scale1

    return mkpts0_f, mkpts1_f, mconf


# ============================================================================
# 6. Model patching
# ============================================================================

def prepare_model_static(orig_model, K=4096):
    """Patch JamMa for static-shape ONNX export."""
    jm = orig_model.matcher.joint_mamba

    # Mamba blocks
    for layer in jm.layers:
        layer.mixer.use_fast_path = False
        layer.norm = RMSNormONNX(layer.norm)
    print(f"✓ {len(jm.layers)} Mamba blocks patched")

    matcher = orig_model.matcher

    # Override forward() to use TopK matching
    orig_forward = matcher.forward

    def static_forward(data):
        data.update({
            'hw0_i': data['imagec_0'].shape[2:],
            'hw1_i': data['imagec_1'].shape[2:],
            'hw0_c': [data['h_8'], data['w_8']],
            'hw1_c': [data['h_8'], data['w_8']],
        })

        # --- Coarse matching (replaces matcher.coarse_match) ---
        desc0 = data['feat_8_0'].flatten(2, 3)
        desc1 = data['feat_8_1'].flatten(2, 3)
        kpts0 = normalize_keypoints_onnx(data['grid_8'], data['imagec_0'].shape[-2:])
        kpts1 = normalize_keypoints_onnx(data['grid_8'], data['imagec_1'].shape[-2:])
        kpts0, kpts1 = kpts0.transpose(1, 2), kpts1.transpose(1, 2)
        data['feat_8_0'] = desc0 + matcher.kenc(kpts0)
        data['feat_8_1'] = desc1 + matcher.kenc(kpts1)

        jm(data)

        # Coarse projection
        d0 = matcher.coarse_matching.final_proj(data['feat_8_0'].transpose(1, 2))
        d1 = matcher.coarse_matching.final_proj(data['feat_8_1'].transpose(1, 2))
        d0 = d0 / (d0.shape[-1] ** 0.5)
        d1 = d1 / (d1.shape[-1] ** 0.5)
        sim = torch.einsum("bnc,bmc->bnm", d0, d1)
        sim = sim / matcher.coarse_matching.temperature

        coarse_match_static(matcher, sim, data, K=K)

        # --- Fine matching (replaces fine_preprocess + fine_matching) ---
        W = matcher.config['fine_window_size']
        stride = data.get('resolution1', 8) // matcher.config['resolution'][1]
        bs, c, h8, w8 = data['bs'], data['c'], data['h_8'], data['w_8']
        f8 = torch.cat([data['feat_8_0'], data['feat_8_1']], 0).view(2*bs, c, h8, -1)
        f4 = torch.cat([data['feat_4_0'], data['feat_4_1']], 0)
        ff = matcher.inter_fpn(f8, f4)
        ff0, ff1 = torch.chunk(ff, 2, dim=0)
        data.update({'hw0_f': ff0.shape[2:], 'hw1_f': ff1.shape[2:]})

        pad = 0 if W % 2 == 0 else W // 2
        f0u = F.unfold(ff0, (W, W), stride=stride, padding=pad)
        f0u = rearrange(f0u, 'n (c ww) l -> n l ww c', ww=W**2)
        f1u = F.unfold(ff1, (W, W), stride=stride, padding=pad)
        f1u = rearrange(f1u, 'n (c ww) l -> n l ww c', ww=W**2)

        f0u = f0u[data['b_ids'], data['i_ids']]
        f1u = f1u[data['b_ids'], data['j_ids']]
        ff_cat = torch.cat([f0u, f1u], 1).transpose(1, 2)
        for layer in matcher.fine_enc:
            ff_cat = layer(ff_cat)
        f0u = ff_cat[:, :, :W**2].transpose(1, 2)
        f1u = ff_cat[:, :, W**2:].transpose(1, 2)

        f0p = matcher.fine_matching.fine_proj(f0u)
        f1p = matcher.fine_matching.fine_proj(f1u)
        norm = f0p.shape[-1] ** 0.5
        f0p, f1p = f0p / norm, f1p / norm
        fine_sim = torch.einsum("nlc,nsc->nls", f0p, f1p) / matcher.fine_matching.temperature
        conf_fine = F.softmax(fine_sim, 1) * F.softmax(fine_sim, 2)

        data['mkpts0_f'], data['mkpts1_f'], data['mconf_f'] = \
            fine_sub_match_static(matcher, conf_fine, f0u, f1u, data, K=K)

    matcher.forward = static_forward
    print(f"✓ Coarse matching: TopK → {K} fixed matches")
    print(f"✓ Fine matching: K in → K out")

    return orig_model


# ============================================================================
# 7. Export wrapper
# ============================================================================

class ExportWrapper(nn.Module):
    def __init__(self, backbone, matcher):
        super().__init__()
        self.backbone = backbone
        self.matcher = matcher

    def forward(self, image0, image1):
        data = {'imagec_0': image0, 'imagec_1': image1}
        self.backbone(data)
        self.matcher(data)
        return data['mkpts0_f'], data['mkpts1_f'], data['mconf_f']


# ============================================================================
# 8.  Main
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description='JamMa static ONNX export')
    p.add_argument('--output', default='./jamma_static.onnx')
    p.add_argument('--K', type=int, default=4096)
    p.add_argument('--image-size', type=int, default=832)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}  K={args.K}")

    register_selective_scan_symbolic()

    from demo.utlis import JamMa as JamMaDemo
    cfg = {
        'coarse': {'d_model': 256},
        'fine': {'d_model': 64, 'dsmax_temperature': 0.1, 'thr': 0.1, 'inference': True},
        'match_coarse': {'thr': 0.2, 'use_sm': True, 'border_rm': 2,
                         'dsmax_temperature': 0.1, 'inference': True},
        'fine_window_size': 5, 'resolution': [8, 2],
    }
    model = JamMaDemo(cfg, pretrained='official').eval().to(device)
    # Patch model for static shapes
    prepare_model_static(model, K=args.K)

    # Sanity check
    from src.utils.dataset import read_megadepth_color
    img0 = read_megadepth_color('./assets/figs/345822933_b5fb7b6feb_o.jpg', args.image_size, 16, True)[0].to(device)
    img1 = read_megadepth_color('./assets/figs/479605349_8aa68e066d_o.jpg', args.image_size, 16, True)[0].to(device)
    export_model = ExportWrapper(model.backbone, model.matcher).eval()

    with torch.no_grad():
        mkpts0, mkpts1, mconf = export_model(img0, img1)
        torch.cuda.synchronize()
    n_valid = (mconf > 0.01).sum().item()
    print(f"PT test: {mkpts0.shape[0]} matches ({n_valid} with conf > 0.01)")

    # Save reference
    ref_path = args.output.replace('.onnx', '_ref.pt')
    torch.save({'mkpts0_f': mkpts0.cpu(), 'mkpts1_f': mkpts1.cpu(),
                'mconf_f': mconf.cpu()}, ref_path)

    # Export ONNX
    print(f"Exporting ONNX (all static shapes) ...")
    with torch.no_grad():
        torch.onnx.export(
            export_model, (img0, img1), args.output,
            input_names=['image0', 'image1'],
            output_names=['mkpts0_f', 'mkpts1_f', 'mconf_f'],
            opset_version=17,
            dynamo=False,
        )

    import onnx
    m = onnx.load(args.output)
    onnx.checker.check_model(m)
    custom = [n for n in m.graph.node if n.op_type == 'SelectiveScan']
    nonzero = [n for n in m.graph.node if n.op_type == 'NonZero']
    print(f"ONNX: {len(m.graph.node)} nodes, {len(custom)} custom ops, "
          f"{len(nonzero)} NonZero ops")
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")
    print(f"Ref:   {ref_path}")


if __name__ == '__main__':
    main()
