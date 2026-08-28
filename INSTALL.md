# Installation Guide (MacOS Intel, Macbook Air 2018)

## Create and enter dir
```bash
mkdir [FOLDER]
cd [FOLDER]
```

# Create and activate environment
We will use uv for speed but not the uv structure (uv.lock, pyproject.toml, etc.), that is more robust but a bit more complex, for simplicity we take this approach.
```bash
uv venv
source .venv/bin/activate
```

# Dependencies
Since we're using MacOS Intel we can install up to certain versions of specific dependencies, for example torch.
```bash
uv pip install "torch==2.2.2" "torchvision>=0.17.2" "torchaudio>=2.2.2"
```

```bash
uv pip install pytorch-lightning pyrender yacs scikit-image einops timm dill roma fvcore trimesh braceexpand loguru optree psutil
```

NOTE: Here we're intentionally omitting `decord`, since we cannot install it. We don't need it though since it is a video processing library, we will only process images. However, SAM may try to import it internally, so we need to mock it.

```bash
from unittest.mock import MagicMock

# TRICK PYTHON INTO IGNORING DECORD
# This creates a fake 'decord' module so the SAM imports don't crash
sys.modules['decord'] = MagicMock()
```

> Before importing SAM!



# Clone the repos into our folder
```bash
git clone https://github.com/facebookresearch/sam-3d-body.git
git clone https://github.com/facebookresearch/sam3.git
```


# Install Detectron2 for CPU only
```bash
FORCE_CUDA="0" uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
```
Here we may run into this error: *No module named 'setuptools'*

The error ModuleNotFoundError  happens because of the `--no-build-isolation` flag we used.

When you tell uv not to use build isolation, it relies entirely on your current virtual environment to compile the package. Newer versions of Python and uv do not install build tools like setuptools or wheel by default to keep the environment lightweight.

We need to manually add the tools required to build C++ extensions from source. Run this in your activated environment:
```bash
uv pip install setuptools wheel ninja
```

It should now successfully find setuptools, build the metadata, and compile the CPU version of Detectron2 for our Mac.

# Install MoGe
```bash
uv pip install git+https://github.com/microsoft/MoGe.git
```

Here we may run into another issue, and is that MoGe requires torch > 2.4, and due to our constraint of having MacOS Intel we can install only up to 2.2.2. Now since SAM uses MoGe basically (among other things probably) to assume an internal virtual camera, usually works fine with slightly older versions of torch, such as ours. So we can forcefully bypass MoGe's strict version check.

Just like we did with Detectron2, you can use the --no-deps (no dependencies) flag. This forces uv to install the MoGe code without trying to resolve or upgrade PyTorch:
```bash
uv pip install git+https://github.com/microsoft/MoGe.git --no-deps
```

# Install sam3 in editable mode
uv pip install -e ./sam3


# After Installation
After we install everything, we need to fix an important issue: Importing from `sam-3d-body`. Since it's name uses hyphens, we cannot directly import. Now we could change the name of the folder, but to not break any internal import or whatsoever, we need to keep it's name untouched. The solution here is to append the path to the system path.

The idea is to do this in our entry point, in our case, the entry point is `main.py`. If we want the entry point to be inside the `app` folder, we would need to do this process in the `__init__.py` file (creating one if we don't have one inside the `app` folder).


# Extra Steps
Once we did the previous steps, we still have some extra things to do:
- Install the dependencies listed in the official SAM repository, in the INSTALL.md
```bash
uv pip install opencv-python pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest black pycocotools tensorboard huggingface_hub
```

We're intentionally excluding `xtcocotools` because we need to have cython and numpy installed.
> The package xtcocotools has an outdated build script that requires numpy to be present in the environment before it can build its Cython C-extensions.
<!-- xtcocotools is an evaluation tool used for COCO keypoint metrics. For running 3D body estimation inference on custom images, it is not required. However, for scalability, we fix the issue. -->
```bash
uv pip install "numpy<2"
```

After this, we can install `xtcocotools`:
```bash
uv pip install xtcocotools --no-build-isolation
```

Since we installed `numpy<2`, we need to downgrade `scipy`, since the latest versions use `numpy>2.x`
```bash
uv pip install "scipy<1.14"
```

Since we downgraded *numpy* and *scipy*, we need to reinstall detectron:
```bash
FORCE_CUDA="0" uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps --force-reinstall
```

Once we've done this, if we try to run our main.py file, we will get an error similar to this:
```python
OSError: ("dlopen(EGL, 0x000A): tried: 'EGL' (no such file), '/System/Volumes/Preboot/Cryptexes/OSEGL' (no such file), '/usr/lib/EGL' (no such file, not in dyld cache), 'EGL' (no such file), '/usr/lib/EGL' (no such file, not in dyld cache)", 'EGL', None)
```

This is happening because pyrender attempts to load EGL (a Linux-only OpenGL backend) when it is imported on macOS. And here we have 2 fixes. Since we need SAM 3D Body to run inference and extract the 3D mesh parameters, we do not need pyrender active, so we can mock it, just as we did with **decord**. 

Alternatively, if we eventually want pyrender to render 3D images on macOS, we must force PyOpenGL to use osmesa or pyglet instead of EGL:
```python
# Entry point
os.environ["PYOPENGL_PLATFORM"] = "pyglet" # or osmesa (we may need to install osmesa, pyglet seems to be already installed)
```
