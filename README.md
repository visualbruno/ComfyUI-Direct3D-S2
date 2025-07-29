# ComfyUI-Direct3D-S2

Requires Python v3.10, v3.11 or v3.12

I created these nodes to refine models generated with Hunyuan 3D v2.0 and v2.1

Use only a "SDF Resolution" of 1024. 512 never generates good results

Optimizations have been done by Easymode-ai. Now it requires about 11Gb of VRAM.

# Known Bug

Removing interior can crash sometimes with OOM error. Try to run again.

# Install requirements

`pip install -r requirements.txt`

# Install voxelize

Go in the folder `voxelize` and run the command: `python setup.py install`

# Install torchsparse

Linux: `pip install torchsparse`

Windows: You will find wheels in the folder `wheels`

# Install Flash Attention

Linux: `pip install flash_attn`

Windows: You can find precompiled wheels here [https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main](https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main)

# Download the models

You will find the models here: [https://huggingface.co/wushuang98/Direct3D-S2/tree/main](https://huggingface.co/wushuang98/Direct3D-S2/tree/main)

Create a folder `wushuang98/Direct3D-S2` in the folder `models` and copy the models `direct3d-s2-v-1-0` and `direct3d-s2-v-1-1` in `wushuang98/Direct3D-S2` folder

You should have a structure like this:

`ComfyUI / models / wushuang98 / Direct3D-S2 / direct3d-s2-v-1-0 / files from huggingface`

`ComfyUI / models / wushuang98 / Direct3D-S2 / direct3d-s2-v-1-1 / files from huggingface`

