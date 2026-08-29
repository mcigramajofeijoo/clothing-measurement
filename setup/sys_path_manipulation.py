import sys
from paths import SAM_3D_BODY_PATH

if str(SAM_3D_BODY_PATH) not in sys.path:
    sys.path.insert(0, str(SAM_3D_BODY_PATH))
