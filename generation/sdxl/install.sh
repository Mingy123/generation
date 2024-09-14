#!/bin/bash

conda create -n sdxlv1 python=3.10 -y
conda activate sdxlv1
pip install diffusers --upgrade
pip install invisible_watermark transformers accelerate safetensors

