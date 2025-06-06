# src/utils.py
import numpy as np

def normalize_obs(joint_angles, end_effector_pos, target_pos):
    """Normaliza las observaciones para el entrenamiento."""
    return np.concatenate([
        joint_angles,
        end_effector_pos,
        target_pos
    ]).astype(np.float32)
