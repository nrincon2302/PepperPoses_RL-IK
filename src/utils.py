# src/utils.py
import numpy as np
# Asegúrate de que StateHistory esté disponible si es necesario,
# pero normalize_obs debería recibir el historial pre-procesado.

def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """Normaliza ángulos a [-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi

def normalize_obs(joint_angles: np.ndarray,
                  end_effector_pos: np.ndarray,
                  target_pos: np.ndarray,
                  history_concatenated: np.ndarray) -> np.ndarray:
    """
    Normaliza y concatena las observaciones.
    history_concatenated es el resultado de history.get_concatenated_history()
    """
    # Normalizar ángulos (opcional, pero puede ayudar al aprendizaje)
    # Si los límites ya son simétricos y no muy grandes, podría no ser necesario.
    # Por ahora, no los normalizaremos aquí, asumiendo que el agente aprende los rangos.
    
    # Normalizar posiciones relativas al efector final podría ser útil,
    # pero por ahora usaremos coordenadas globales.

    obs = np.concatenate([
        joint_angles.astype(np.float32),
        end_effector_pos.astype(np.float32),
        target_pos.astype(np.float32),
        history_concatenated.astype(np.float32)  # Ya viene pre-procesado y aplanado
    ])
    return obs
    