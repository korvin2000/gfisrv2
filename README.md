# Gated Fourier Inception Super Resolution v2

GFISRv2 is an experimental super-resolution architecture that augments
Inception-style branches with a learnable FFT block. The goal is to
strengthen global context modeling while remaining efficient and easy to
integrate into existing PyTorch projects.

## Introduction

It has been a while since new architectures or models were released for this
project. This iteration completes a long-running exploration of FFT-based
models that began almost a year ago, when there were no stable, high-quality
implementations available.

## What's New

After a series of experiments, an FFT module was integrated into one of the
Inception branches. This block boosts the model's ability to capture global
context, improving the overall quality of super-resolution outputs.

## Compatibility

To the best of our knowledge, this is the only FFT-based Inception module that
is fully compatible with exporting from PyTorch to ONNX. The current
implementation targets ONNX opset 17.

## Usage

```python
from gfisrv2 import GFISRV2

model = GFISRV2()
# proceed with fine-tuning or inference
```

## Acknowledgements

This repository builds on many ideas from the PyTorch and ONNX communities and
draws inspiration from recent research into Fourier transform-based models.

