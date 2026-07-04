#!/usr/bin/env python3
"""
JamMa TRT Engine — Inference
==============================

Usage:
    python run_trt_inference.py --image0 a.jpg --image1 b.jpg
    python run_trt_inference.py --image0 a.jpg --image1 b.jpg --size 512
"""

import os, sys, time, ctypes, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description='JamMa TRT Inference')
    p.add_argument('--image0', default='assets/figs/345822933_b5fb7b6feb_o.jpg', help='First image path')
    p.add_argument('--image1', default='assets/figs/479605349_8aa68e066d_o.jpg', help='Second image path')
    p.add_argument('--engine', default='jamma_full_fp32.engine', help='TRT engine path')
    p.add_argument('--size', type=int, default=832, help='Resize longer side to this')
    p.add_argument('--visualize', default='output_trt/trt_matches.png',
                   help='Save match visualization (default: output_trt/trt_matches.png)')
    p.add_argument('--benchmark', type=int, default=20, help='Benchmark iterations (0=skip)')
    p.add_argument('--compare', action='store_true',
                   help='Run both PT and TRT, compare coordinate accuracy')
    return p.parse_args()


def load_image(path, size):
    """Preprocess exactly as ONNX export does: use read_megadepth_color."""
    from src.utils.dataset import read_megadepth_color
    return read_megadepth_color(path, size, 16, True)[0]


def main():
    args = parse_args()

    # Auto-derive engine path
    engine_path = args.engine
    if engine_path is None:
        candidates = [
            f'./jamma_static_{args.size}x{args.size}.engine',
            './jamma_full_fp32.engine',
        ]
        for c in candidates:
            if Path(c).exists():
                engine_path = c
                break
    if engine_path is None or not Path(engine_path).exists():
        sys.exit(f'Engine not found. Build it: python build_trt_engine.py --size {args.size}')

    # Load plugin
    plugin = './trt_plugin/build/libjam_plugin.so'
    if Path(plugin).exists():
        ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)

    # Load & preprocess images (same as ONNX export)
    print(f'Loading: {args.image0}, {args.image1}')
    img0 = load_image(args.image0, args.size).cuda()
    img1 = load_image(args.image1, args.size).cuda()
    print(f'Input size: {img0.shape[2]}×{img0.shape[3]}')

    # Load engine
    import tensorrt as trt
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Allocate I/O
    buf = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name in ('image0', 'image1'):
            continue
        shape = tuple(ctx.get_tensor_shape(name))
        buf[name] = torch.empty(shape, dtype=torch.float32, device='cuda')

    ctx.set_input_shape('image0', img0.shape)
    ctx.set_input_shape('image1', img1.shape)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name == 'image0':
            ctx.set_tensor_address(name, img0.data_ptr())
        elif name == 'image1':
            ctx.set_tensor_address(name, img1.data_ptr())
        else:
            ctx.set_tensor_address(name, buf[name].data_ptr())

    # Warmup
    for _ in range(3):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Benchmark (optional)
    if args.benchmark > 0:
        times = []
        for _ in range(args.benchmark):
            t0 = time.perf_counter()
            ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        print(f'Latency: {np.mean(times):.0f} ms (±{np.std(times):.0f}) '
              f'over {args.benchmark} runs')

    # Run inference
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Results
    mkpts0 = buf['mkpts0_f'].cpu().numpy()
    mkpts1 = buf['mkpts1_f'].cpu().numpy()
    mconf  = buf['mconf_f'].cpu().numpy()
    valid  = mconf > 0.01

    print(f'Matches: {valid.sum()} valid / {len(mconf)} total')
    print(f'Confidence range: [{mconf[valid].min():.3f}, {mconf[valid].max():.3f}] '
          f'(median: {np.median(mconf[valid]):.3f})')

    # --compare: run PyTorch model on same images, diff coordinates
    if args.compare:
        print(f'\n--- PT vs TRT Comparison ---')
        from export_jamma_onnx_static import (
            register_selective_scan_symbolic, prepare_model_static, ExportWrapper)
        register_selective_scan_symbolic()
        from demo.utlis import JamMa as JamMaDemo
        cfg = {
            'coarse': {'d_model': 256},
            'fine': {'d_model': 64, 'dsmax_temperature': 0.1, 'thr': 0.1, 'inference': True},
            'match_coarse': {'thr': 0.2, 'use_sm': True, 'border_rm': 2,
                             'dsmax_temperature': 0.1, 'inference': True},
            'fine_window_size': 5, 'resolution': [8, 2],
        }
        model = JamMaDemo(cfg, pretrained='official').eval().cuda()
        prepare_model_static(model, K=8000)
        pt_model = ExportWrapper(model.backbone, model.matcher).eval()
        with torch.no_grad():
            pt0, pt1, ptc = pt_model(img0, img1)
            torch.cuda.synchronize()
        pt0, pt1, ptc = pt0.cpu().numpy(), pt1.cpu().numpy(), ptc.cpu().numpy()

        n_pt = (ptc > 0.01).sum()
        n_trt = valid.sum()
        both = (ptc > 0.01) & (mconf > 0.01)
        if both.sum() > 0:
            # Use KD-tree: find nearest TRT match for each PT match
            from scipy.spatial import KDTree
            tree0 = KDTree(mkpts0[mconf > 0.01]); tree1 = KDTree(mkpts1[mconf > 0.01])
            d0, i0 = tree0.query(pt0[ptc > 0.01])
            d1, i1 = tree1.query(pt1[ptc > 0.01])
            same = i0 == i1
            match_d = np.maximum(d0, d1)
            print(f'  PT matches: {n_pt}  TRT matches: {n_trt}')
            print(f'  Coord diff (KD-tree matched): '
                  f'max={match_d.max():.1f} px  median={np.median(match_d):.4f} px')
            for thr in [0.1, 0.5, 1.0]:
                n = (match_d < thr).sum()
                print(f'  <{thr:.1f} px: {n}/{n_pt} ({100*n/n_pt:.0f}%)')
            print(f'  Same-index (matched to same TRT pt): {same.sum()}/{n_pt}')
            for thr in [0.1, 0.5, 1.0]:
                n = (match_d < thr).sum()
                print(f'  <{thr:.1f} px: {n}/{both.sum()} ({100*n/both.sum():.0f}%)')
        del model, pt_model; torch.cuda.empty_cache()
    print(f'Confidence range: [{mconf[valid].min():.3f}, {mconf[valid].max():.3f}] '
          f'(median: {np.median(mconf[valid]):.3f})')

    # Visualize
    if args.visualize:
        Path(args.visualize).parent.mkdir(parents=True, exist_ok=True)
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        im0 = img0[0].permute(1, 2, 0).cpu().numpy()
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        im0 = (im0 * std + mean).clip(0, 1)
        im1 = img1[0].permute(1, 2, 0).cpu().numpy()
        im1 = (im1 * std + mean).clip(0, 1)
        H, W = im0.shape[:2]

        canvas = np.ones((H, W * 2, 3), dtype=np.float32)
        canvas[:, :W] = im0; canvas[:, W:] = im1

        pts0, pts1, conf = mkpts0[valid][:300], mkpts1[valid][:300], mconf[valid][:300]
        fig, ax = plt.subplots(figsize=(22, 13), dpi=150)
        ax.imshow(canvas); ax.axis('off')
        norm = Normalize(vmin=conf.min(), vmax=conf.max())
        cmap = plt.get_cmap('viridis')
        for i in range(len(pts0)):
            c = cmap(norm(conf[i]))
            ax.plot(pts0[i, 0], pts0[i, 1], 'o', color=c, ms=1, alpha=0.6)
            ax.plot(pts1[i, 0] + W, pts1[i, 1], 'o', color=c, ms=1, alpha=0.6)
            ax.plot([pts0[i, 0], pts1[i, 0] + W],
                    [pts0[i, 1], pts1[i, 1]], '-', color=c, lw=0.2, alpha=0.35)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02).set_label('Confidence', fontsize=10)
        plt.title(f'JamMa TRT — {valid.sum()} matches', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(args.visualize, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Visualization: {args.visualize}')


if __name__ == '__main__':
    main()
