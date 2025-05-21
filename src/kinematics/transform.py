import numpy as np
from scipy.spatial.transform import Rotation as R

def homogeneous_transform(angle: float, axis: str, translate: np.ndarray=None) -> np.ndarray:
    T = np.eye(4)
    if axis in ('x','y','z'):
        T[:3,:3] = R.from_euler(axis, angle).as_matrix()
    else:
        raise ValueError(f"Unknown axis {axis}")
    if translate is not None:
        T[:3,3] = translate
    return T