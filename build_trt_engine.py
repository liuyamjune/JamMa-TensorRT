#!/usr/bin/env python3
"""
JamMa: Build TRT Engine from PyTorch Model
============================================

One script, full pipeline:
  1. Export full-model ONNX with static shapes (nonzero+pad, SelectiveScan custom op)
  2. Load pre-compiled TRT Plugin (libjam_plugin.so)
  3. Parse ONNX → TRT network, build engine, save to disk
  4. Verify accuracy against PyTorch reference
  5. Benchmark: latency, throughput, GPU memory

Prerequisites:
    cd trt_plugin/build && cmake .. -DTRT_ROOT=/usr/local/TensorRT-* && make -j
    → produces libjam_plugin.so

Usage:
    conda activate jamma
    python build_trt_engine.py                    # 832x832 FP32
    python build_trt_engine.py --size 640         # 640x640 FP32
    python build_trt_engine.py --size 512         # 512x512 FP32
    python build_trt_engine.py --size 832 --fp16  # 832x832 FP16 (⚠ accuracy)

Output:
    jamma_full_fp32.engine    # TRT engine
    jamma_static_ref.pt       # PyTorch reference for accuracy comparison
"""

import os, sys, time, ctypes, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# Command-line
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='JamMa TRT Engine Builder')
    p.add_argument('--size', type=int, default=832, help='Input image size (square)')
    p.add_argument('--K', type=int, default=8000, help='Fixed number of output matches')
    p.add_argument('--fp16', action='store_true', help='Enable FP16 (⚠ reduces accuracy)')
    p.add_argument('--workspace-gb', type=int, default=4, help='TRT workspace in GiB')
    p.add_argument('--plugin', default='./trt_plugin/build/libjam_plugin.so')
    p.add_argument('--output-dir', default='.', help='Output directory for engine + ONNX')
    p.add_argument('--image0', default='assets/figs/345822933_b5fb7b6feb_o.jpg')
    p.add_argument('--image1', default='assets/figs/479605349_8aa68e066d_o.jpg')
    p.add_argument('--benchmark-iters', type=int, default=30)
    p.add_argument('--skip-verify', action='store_true')
    p.add_argument('--skip-benchmark', action='store_true')
    return p.parse_args()


# ===========================================================================
# Step 1: Export ONNX
# ===========================================================================

def export_onnx(size, K, output_dir):
    """Export full JamMa ONNX with static shapes + custom SelectiveScan op."""
    from export_jamma_onnx_static import (
        register_selective_scan_symbolic,
        prepare_model_static,
        ExportWrapper,
    )
    from demo.utlis import JamMa as JamMaDemo
    from src.utils.dataset import read_megadepth_color

    register_selective_scan_symbolic()

    cfg = {
        'coarse': {'d_model': 256},
        'fine': {'d_model': 64, 'dsmax_temperature': 0.1, 'thr': 0.1,
                 'inference': True},
        'match_coarse': {'thr': 0.2, 'use_sm': True, 'border_rm': 2,
                         'dsmax_temperature': 0.1, 'inference': True},
        'fine_window_size': 5, 'resolution': [8, 2],
    }

    onnx_path = os.path.join(output_dir, f'jamma_static_{size}x{size}.onnx')
    ref_path   = os.path.join(output_dir, f'jamma_static_{size}x{size}_ref.pt')

    print(f'[1/4] Exporting ONNX ({size}x{size}, K={K}) ...')

    model = JamMaDemo(cfg, pretrained='official').eval().cuda()
    prepare_model_static(model, K=K)

    img0 = read_megadepth_color(args.image0, size, 16, True)[0].cuda()
    img1 = read_megadepth_color(args.image1, size, 16, True)[0].cuda()

    export_model = ExportWrapper(model.backbone, model.matcher).eval()
    with torch.no_grad():
        mkpts0, mkpts1, mconf = export_model(img0, img1)
        torch.cuda.synchronize()

    n_valid = (mconf > 0.01).sum().item()
    print(f'  PT reference: {n_valid} valid matches')
    torch.save({
        'mkpts0_f': mkpts0.cpu(),
        'mkpts1_f': mkpts1.cpu(),
        'mconf_f': mconf.cpu(),
    }, ref_path)

    with torch.no_grad():
        torch.onnx.export(
            export_model, (img0, img1), onnx_path,
            input_names=['image0', 'image1'],
            output_names=['mkpts0_f', 'mkpts1_f', 'mconf_f'],
            opset_version=17, dynamo=False,
        )

    import onnx
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    custom_ops = [n for n in m.graph.node
                  if n.op_type == 'SelectiveScan' and n.domain == 'jam']
    nonzero_ops = [n for n in m.graph.node if n.op_type == 'NonZero']
    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f'  ONNX: {len(m.graph.node)} nodes, '
          f'{len(custom_ops)} SelectiveScan, {len(nonzero_ops)} NonZero, '
          f'{size_mb:.1f} MB')

    del model, export_model
    torch.cuda.empty_cache()
    return onnx_path, ref_path


# ===========================================================================
# Step 2: Load plugin
# ===========================================================================

def load_plugin(plugin_path):
    if not Path(plugin_path).exists():
        sys.exit(f'Plugin not found: {plugin_path}\n'
                 f'Run: cd trt_plugin/build && cmake .. && make -j')
    ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
    print(f'[2/4] Plugin loaded: {plugin_path}')


# ===========================================================================
# Step 3: Build TRT engine
# ===========================================================================

def build_engine(onnx_path, size, fp16, workspace_gb, output_dir):
    import tensorrt as trt

    precision = 'FP16' if fp16 else 'FP32'
    engine_path = onnx_path.replace('.onnx', f'_{precision.lower()}.engine')
    if not fp16:
        engine_path = onnx_path.replace('.onnx', '.engine')  # default name
    # For 832, use simpler name
    if size == 832 and not fp16:
        engine_path = os.path.join(output_dir, 'jamma_full_fp32.engine')

    print(f'[3/4] Building TRT engine ({size}x{size}, {precision}) ...')

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * 1024**3)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f'  FP16 enabled (⚠ may reduce accuracy)')

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(min(5, parser.num_errors)):
                print(f'  PARSE ERROR: {parser.get_error(i)}')
            sys.exit(1)
    print(f'  Parsed: {network.num_layers} layers')

    profile = builder.create_optimization_profile()
    profile.set_shape('image0', (1, 3, size, size),
                      (1, 3, size, size), (1, 3, size, size))
    profile.set_shape('image1', (1, 3, size, size),
                      (1, 3, size, size), (1, 3, size, size))
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit('  FAILED: build_serialized_network returned None')

    elapsed = time.perf_counter() - t0
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / 1024**2
    print(f'  Engine: {engine_path} ({size_mb:.1f} MB, {elapsed:.0f}s)')
    return engine_path


# ===========================================================================
# Step 4: Verify + Benchmark
# ===========================================================================

def verify_and_benchmark(engine_path, ref_path, size, iters):
    import tensorrt as trt
    from src.utils.dataset import read_megadepth_color
    import ctypes as ct

    print(f'[4/4] Verification & Benchmark ...')

    img0 = read_megadepth_color(args.image0, size, 16, True)[0].cuda()
    img1 = read_megadepth_color(args.image1, size, 16, True)[0].cuda()

    # Measure GPU memory BEFORE loading engine
    cuda = ct.CDLL("libcudart.so")
    free_empty = ct.c_longlong(); total = ct.c_longlong()
    cuda.cudaMemGetInfo(ct.byref(free_empty), ct.byref(total))

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # Allocate buffers
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
    for _ in range(5):
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Measure GPU memory after engine + IO buffers loaded
    free_loaded = ct.c_longlong()
    cuda.cudaMemGetInfo(ct.byref(free_loaded), ct.byref(total))
    vram_used = (free_empty.value - free_loaded.value) / 1024**2

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    # Results
    mconf = buf['mconf_f'].cpu().numpy()
    n_valid = (mconf > 0.01).sum()

    print(f'\n{"="*55}')
    print(f'  TRT Engine — {size}×{size} FP32')
    print(f'{"="*55}')
    print(f'  Latency:     {np.mean(times):.0f} ms (±{np.std(times):.0f})')
    print(f'  Throughput:  {1000/np.mean(times):.1f} img/s')
    print(f'  GPU memory:  {vram_used:.0f} MiB (total GPU: {total.value/1024**3:.1f} GB)')
    print(f'  Matches:     {n_valid} valid / {len(mconf)} total')
    print(f'  Engine:      {engine_path}')

    # Accuracy
    if not args.skip_verify and Path(ref_path).exists():
        ref = torch.load(ref_path)
        r0 = ref['mkpts0_f'].numpy(); r1 = ref['mkpts1_f'].numpy()
        rc = ref['mconf_f'].numpy()
        t0_np = buf['mkpts0_f'].cpu().numpy(); t1_np = buf['mkpts1_f'].cpu().numpy()

        # KD-tree match: for each valid PT match, find nearest TRT match
        from scipy.spatial import KDTree
        trt_valid = mconf > 0.01; pt_valid = rc > 0.01
        tree0 = KDTree(t0_np[trt_valid]); tree1 = KDTree(t1_np[trt_valid])
        d0, i0 = tree0.query(r0[pt_valid]); d1, i1 = tree1.query(r1[pt_valid])
        match_d = np.maximum(d0, d1); same = i0 == i1

        print(f'\n  Accuracy vs PyTorch reference (KD-tree matched):')
        print(f'    PT matches: {(rc>0.01).sum()}  TRT matches: {n_valid}')
        print(f'    Coord error:   max={match_d.max():.1f} px  '
              f'median={np.median(match_d):.4f} px  mean={match_d.mean():.3f} px')
        for thr in [0.1, 0.5, 1.0]:
            n = (match_d < thr).sum()
            print(f'    <{thr:.1f} px:       {n}/{(rc>0.01).sum()} ({100*n/(rc>0.01).sum():.0f}%)')
        print(f'    Same-index:     {same.sum()}/{(rc>0.01).sum()}')

    print(f'{"="*55}\n')

    # Visualization
    viz_path = engine_path.replace('.engine', '_matches.png')
    visualize_matches(buf, img0, img1, viz_path)
    print(f'  Visualization: {viz_path}')


# ===========================================================================
# Visualization helper
# ===========================================================================

def visualize_matches(buf, img0, img1, output_path, n_show=300):
    """Draw top-N matches as a side-by-side image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    t0 = buf['mkpts0_f'].cpu().numpy()
    t1 = buf['mkpts1_f'].cpu().numpy()
    tc = buf['mconf_f'].cpu().numpy()
    valid = tc > 0.01
    mkpts0, mkpts1, mconf = t0[valid], t1[valid], tc[valid]

    if len(mkpts0) == 0:
        print('  ⚠ No valid matches to visualize')
        return

    # Reconstruct padded input images from tensors
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    im0 = (img0[0].permute(1,2,0).cpu().numpy() * std + mean).clip(0, 1)
    im1 = (img1[0].permute(1,2,0).cpu().numpy() * std + mean).clip(0, 1)
    H, W = im0.shape[:2]

    canvas = np.ones((H, W * 2, 3), dtype=np.float32)
    canvas[:, :W] = im0
    canvas[:, W:] = im1

    n = min(n_show, len(mkpts0))
    idx = np.argsort(-mconf)[:n]
    mkpts0, mkpts1, mconf = mkpts0[idx], mkpts1[idx], mconf[idx]

    fig, ax = plt.subplots(figsize=(22, 13), dpi=150)
    ax.imshow(canvas); ax.axis('off')

    norm = Normalize(vmin=mconf.min(), vmax=mconf.max())
    cmap = plt.get_cmap('viridis')
    for i in range(n):
        c = cmap(norm(mconf[i]))
        ax.plot(mkpts0[i,0], mkpts0[i,1], 'o', color=c, ms=1, alpha=0.6)
        ax.plot(mkpts1[i,0]+W, mkpts1[i,1], 'o', color=c, ms=1, alpha=0.6)
        ax.plot([mkpts0[i,0], mkpts1[i,0]+W],
                [mkpts0[i,1], mkpts1[i,1]], '-', color=c, lw=0.2, alpha=0.35)

    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.02).set_label('Confidence', fontsize=10)
    plt.title(f'JamMa TRT Engine — {len(mkpts0)} matches (top {n})',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    args = parse_args()

    t_total = time.perf_counter()

    load_plugin(args.plugin)
    onnx_path, ref_path = export_onnx(args.size, args.K, args.output_dir)
    engine_path = build_engine(onnx_path, args.size, args.fp16,
                               args.workspace_gb, args.output_dir)
    verify_and_benchmark(engine_path, ref_path, args.size,
                        args.benchmark_iters)

    print(f'Total time: {time.perf_counter() - t_total:.0f}s')
    print(f'Engine: {engine_path}')
