# src/memory/history.py
import numpy as np

class StateHistory:
    def __init__(self, size: int, dim_joints: int, dim_eff_pos: int = 3):
        self.size = size
        self.dim_joints = dim_joints
        self.dim_eff_pos = dim_eff_pos
        # Historial de ángulos articulares y posiciones del efector final
        # Se almacenan como una lista de np.ndarrays
        self.joints_history = []
        self.positions_history = []

    def clear(self):
        self.joints_history.clear()
        self.positions_history.clear()

    def append(self, joint_angles: np.ndarray, eff_pos: np.ndarray):
        self.joints_history.append(joint_angles.copy())
        self.positions_history.append(eff_pos.copy())
        if len(self.joints_history) > self.size:
            self.joints_history.pop(0)
            self.positions_history.pop(0)

    def get_concatenated_history(self) -> np.ndarray:
        """
        Retorna un array aplanado de los (size-1) estados históricos más recientes.
        Si el historial no está lleno, rellena con ceros.
        El estado actual NO se incluye aquí, se asume que se concatena por separado.
        """
        hist_data = []
        # Iteramos desde el penúltimo hacia atrás, hasta size-1 elementos
        num_hist_states_to_get = self.size - 1
        
        # Llenamos con los estados históricos disponibles (excluyendo el más reciente, que es el "actual")
        # El historial se almacena [oldest, ..., newest_minus_1, newest_current_in_env]
        # Queremos [oldest, ..., newest_minus_1]
        
        # Si self.joints_history tiene N elementos, y queremos M = size-1 estados históricos:
        # Si N < M+1, significa que no tenemos suficientes estados almacenados (incluyendo el actual).
        #   Tomamos todos los que hay excepto el último.
        # Si N = M+1, tenemos exactamente los necesarios. Tomamos todos menos el último.

        actual_available_historical_states = max(0, len(self.joints_history) - 1)
        
        for i in range(actual_available_historical_states):
            hist_data.extend(self.joints_history[i])
            hist_data.extend(self.positions_history[i])
            
        # Rellenar con ceros si no hay suficientes estados históricos
        expected_len = num_hist_states_to_get * (self.dim_joints + self.dim_eff_pos)
        padding_len = expected_len - len(hist_data)
        
        if padding_len > 0:
            hist_data.extend(np.zeros(padding_len, dtype=np.float32))
            
        return np.array(hist_data, dtype=np.float32)

    def latest_violation(self, joint_angles: np.ndarray) -> bool:
        # Esta función no es necesaria si el clipping se maneja en el step
        # y la penalización por límite se aplica allí.
        # Para simplificar, la dejamos retornando False.
        # Podría implementarse para detectar oscilaciones en los límites si fuera necesario.
        return False