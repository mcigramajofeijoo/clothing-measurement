# Installation Guide (Google Colab)

```bash
!git clone -b colab --single-branch --recurse-submodules https://github.com/mcigramajofeijoo/clothing-measurement.git
```

```bash
%cd clothing-measurement/
```

```bash
!pip install pytorch-lightning pyrender yacs scikit-image einops timm dill roma fvcore trimesh braceexpand loguru optree psutil decord
```

```bash
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

```bash
!pip install git+https://github.com/microsoft/MoGe.git
```

```bash
%cd sam3
!pip install -e .
%cd ..
```

```bash
# Why We're Doing This:
# Colab (and moge) natively uses the newer numpy>=2.0. However, the sam3 repository has an outdated strict requirement in its setup files that forces Python to downgrade to numpy 1.26.4. That downgrade instantly broke all of Colab's pre-installed data science packages and moge, while also installing an iopath version that detectron2 doesn't like.
# To fix this, we just need to force numpy back up and manually pin iopath to a version Detectron2 accepts.

!pip install "numpy>=2.0" "iopath<0.1.10"

# • In the top menu of Colab, click Runtime > Restart session (or Restart runtime).
# • Do NOT run the installation cells again.
# • If pip throws a red warning saying SAM3 requires 'numpy<2', you can safely ignore it. SAM3 works perfectly fine with 'numpy>=2.x', it's just a strict version lock left by the developers.
```

```bash
!pip install opencv-python pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest black pycocotools tensorboard huggingface_hub
```

```bash
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN')
```
login(hf_token)

```bash
!pip install ultralytics
```

```bash
# Make sure we're inside our folder!
!touch setup/__init__.py # If not for some reason it cannot read our "setup/sys_path_manipulation.py"
```