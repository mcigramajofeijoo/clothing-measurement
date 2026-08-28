import os
import sys
from unittest.mock import MagicMock
import warnings

# INFO: `UserWarning: Momentum is not enabled:` This warning only matters during model training and has no effect on mesh accuracy, speed, or output quality.
warnings.filterwarnings("ignore", category=UserWarning) 

# Mock decord globally
sys.modules['decord'] = MagicMock()

os.environ["PYOPENGL_PLATFORM"] = "pyglet"

# 2. Add sam-3d-body to sys.path globally
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_BODY_PATH = os.path.join(CURRENT_DIR, "sam-3d-body")
if SAM_BODY_PATH not in sys.path:
    sys.path.append(SAM_BODY_PATH)

from notebook.utils import setup_sam_3d_body

if __name__ == "__main__":
    print("Testing")