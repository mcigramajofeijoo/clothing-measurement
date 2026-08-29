import zipfile
import tempfile
import torch
import os

# Force PyTorch to pretend CUDA is available/dummy-initialized if queried
if hasattr(torch.cuda, "current_device"):
    torch.cuda.current_device = lambda: 0

# Intercept Lightning Fabric's device mixin property safely
try:
    from lightning_fabric.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
    DeviceDtypeModuleMixin.device = property(lambda self: torch.device("cpu"))
except (ImportError, AttributeError):
    pass

# --- PyTorch 2.2.2 Compatibility Patch for DINOv3 & Roma ---
if not hasattr(torch.amp, "custom_fwd"):
    def _compat_custom_fwd(fwd=None, **kwargs):
        if fwd is not None:
            return torch.cuda.amp.custom_fwd(fwd)
        return lambda fn: torch.cuda.amp.custom_fwd(fn)

    def _compat_custom_bwd(bwd=None, **kwargs):
        if bwd is not None:
            return torch.cuda.amp.custom_bwd(bwd)
        return lambda fn: torch.cuda.amp.custom_bwd(fn)

    torch.amp.custom_fwd = _compat_custom_fwd
    torch.amp.custom_bwd = _compat_custom_bwd

# 2. Compatibility Patch for JIT Unpickler UInt32Storage (C++ Level)
_orig_jit_load = torch.jit.load

def _patched_jit_load(f, *args, **kwargs):
    if isinstance(f, str) and f.endswith(".pt"):
        # Save to a temp directory to avoid Hugging Face cache permission issues
        patched_f = os.path.join(tempfile.gettempdir(), os.path.basename(f) + ".patched.pt")
        
        if not os.path.exists(patched_f):
            try:
                with zipfile.ZipFile(f, "r") as zin:
                    with zipfile.ZipFile(patched_f, "w") as zout:
                        for item in zin.infolist():
                            data = zin.read(item.filename)
                            if item.filename.endswith(".pkl"):
                                # Swap unsupported UInt32 for universally supported Int32 (both 4 bytes)
                                data = data.replace(b"UInt32Storage", b"IntStorage")
                            zout.writestr(item.filename, data)
            except Exception:
                pass # Fallback to original if file isn't a standard zip
        
        if os.path.exists(patched_f):
            f = patched_f
            
    return _orig_jit_load(f, *args, **kwargs)

torch.jit.load = _patched_jit_load

# 3. CPU-Only Force Patch (Intercepts hardcoded .to("cuda") calls)
def _remap_to_cpu(device_arg):
    if isinstance(device_arg, str) and "cuda" in device_arg:
        return "cpu"
    if isinstance(device_arg, torch.device) and device_arg.type == "cuda":
        return torch.device("cpu")
    return device_arg

_orig_nn_to = torch.nn.Module.to
def _patched_nn_to(self, *args, **kwargs):
    args = tuple(_remap_to_cpu(a) for a in args)
    if "device" in kwargs:
        kwargs["device"] = _remap_to_cpu(kwargs["device"])
    return _orig_nn_to(self, *args, **kwargs)
torch.nn.Module.to = _patched_nn_to

_orig_tensor_to = torch.Tensor.to
def _patched_tensor_to(self, *args, **kwargs):
    args = tuple(_remap_to_cpu(a) for a in args)
    if "device" in kwargs:
        kwargs["device"] = _remap_to_cpu(kwargs["device"])
    return _orig_tensor_to(self, *args, **kwargs)
torch.Tensor.to = _patched_tensor_to
