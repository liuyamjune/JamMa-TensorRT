# JamMa-TensorRT
origin links:
[![license](https://img.shields.io/badge/LICENSE-MIT-green)](https://github.com/leoluxxx/JamMa/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.03437-red)](https://arxiv.org/abs/2503.03437)
[![webpage](https://img.shields.io/badge/Webpage-JamMa-blue)](https://leoluxxx.github.io/JamMa-page/)
[![star](https://img.shields.io/github/stars/leoluxxx/JamMa)](https://github.com/leoluxxx/JamMa)

This is a PyTorch implementation of JamMa for CVPR'25 [paper](https://arxiv.org/abs/2503.03437)  “JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba”.

JamMa is a lightweight image matcher that enables fast internal and mutual interaction of images with joint Mamba.

## Start to export 
Running this Python script produces both an ONNX model and an engine model.
The selective_scan CUDA kernel only supports FP32, so only FP32 mode is available.
```bash
python build_trt_engine.py
```
## Evaluation
Processing two 832×832 images on an Nvidia RTX 5060 Laptop takes approximately 160ms.
The engine model achieves accuracy on par with the original PyTorch model.

```md
  PT matches: 5666  TRT matches: 5660
  Coord diff (KD-tree matched): max=6.4 px  median=0.0007 px
  <0.1 px: 5625/5666 (99%)
  <0.5 px: 5627/5666 (99%)
  <1.0 px: 5627/5666 (99%)
  Same-index (matched to same TRT pt): 5659/5666
  <0.1 px: 5625/5660 (99%)
  <0.5 px: 5627/5660 (99%)
  <1.0 px: 5627/5660 (99%)
Confidence range: [0.102, 0.987] (median: 0.687)
```

```
@misc{lu2025jamma,
    title={JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba},
    author={Xiaoyong Lu and Songlin Du},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
}
```
