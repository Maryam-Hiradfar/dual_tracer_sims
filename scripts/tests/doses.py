"quick check for doses simulated"
import numpy as np
import sys
sys.path.append("..")
from tracers.aif import pbr28_aif
from tracers.aif import estimated_injected_dose
from tracers.aif import feng_aif
#------------------------------------
# Config
#------------------------------------
Blood_volume_ml = 5000
PBR_scale = 1.0 
fdg_scale = 1.0


print("[INFO] Using blood volume (mL):", Blood_volume_ml)
print("[INFO] Using PBR28 scale factor:", PBR_scale)
print("[INFO] Estimated injected dose (kBq, mCi):", estimated_injected_dose(lambda t: pbr28_aif(t, scale = PBR_scale), blood_volume_ml = Blood_volume_ml))

print("[INFO] Using FDG scale factor:", fdg_scale)
print("[INFO] Estimated injected dose (kBq, mCi):", estimated_injected_dose(lambda t: feng_aif( 20, scale = fdg_scale), blood_volume_ml = Blood_volume_ml))


