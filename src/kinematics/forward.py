import numpy as np
from .transform import homogeneous_transform
from typing import Tuple, List

def forward_kinematics(angles: np.ndarray, params: dict) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Devuelve posición del efector y lista de frames intermedios."""
    j1,j2,j3,j4,j5 = angles
    p = params
    # Base->shoulder offset
    T = homogeneous_transform(0,'x', [0,-p['shoulder_offset_y'],0])
    frames = [T.copy()]
    # Secuencia: pitch(Y), roll(Z), link1, elbow offset, yaw(Z), roll(X), link2, wrist yaw(X)
    for angle, axis, translate in [
        (j1,'y',None), (j2,'z',None), (0,'x',[p['upper_arm_length'],0,0]),
        (0,'x',[0,p['elbow_offset_y'],0]), (j3,'z',None),
        (j4,'x',None), (0,'x',[p['lower_arm_length'],0,0]), (j5,'x',None)
    ]:
        T = T @ homogeneous_transform(angle, axis, translate)
        frames.append(T.copy())
    ee_pos = frames[-1][:3,3]
    return ee_pos, frames