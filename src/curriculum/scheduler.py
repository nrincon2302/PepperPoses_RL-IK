# src/curriculum/scheduler.py
import numpy as np

class CurriculumScheduler:
    """
    Controla niveles de currículo para rango de ángulos y generación de objetivos.
    """
    def __init__(self, num_curriculum_levels: int, initial_angle_factor=0.1, initial_target_factor=0.2, min_target_dist=0.1):
        """
        Args:
            num_curriculum_levels: Número total de niveles discretos en el currículo.
            initial_angle_factor: Factor inicial para la amplitud de los rangos angulares (0 a 1).
                                  1 significa rango completo, 0.1 significa 10% del rango.
            initial_target_factor: Factor inicial para la distancia del objetivo (0 a 1).
                                   1 significa alcance completo, 0.2 significa 20% del alcance.
            min_target_dist: Mínima distancia al objetivo.
        """
        self.num_curriculum_levels = num_curriculum_levels
        self.level = 0  # Nivel actual, de 0 a num_curriculum_levels-1
        
        self.initial_angle_factor = initial_angle_factor
        self.initial_target_factor = initial_target_factor
        self.min_target_dist = min_target_dist

    def get_progress_fraction(self) -> float:
        """Retorna la fracción de progreso en el currículo (0 a 1)."""
        if self.num_curriculum_levels <= 1:
            return 1.0 # Currículo completo si solo hay un nivel o menos
        return min(self.level / (self.num_curriculum_levels -1), 1.0)

    def get_ranges(self, joint_limits_low: np.ndarray, joint_limits_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Obtiene los rangos de ángulos para el estado inicial según el nivel actual.
        A medida que el currículo avanza, los rangos se expanden.
        """
        progress = self.get_progress_fraction()
        
        # Interpolar el factor de amplitud del rango
        # Empieza en initial_angle_factor y va hacia 1.0
        current_angle_amplitude_factor = self.initial_angle_factor + progress * (1.0 - self.initial_angle_factor)
        
        # Calcular el centro de los límites originales
        center = (joint_limits_low + joint_limits_high) / 2.0
        original_half_span = (joint_limits_high - joint_limits_low) / 2.0
        
        # Nuevo semi-rango basado en el factor de amplitud
        current_half_span = original_half_span * current_angle_amplitude_factor
        
        low = center - current_half_span
        high = center + current_half_span
        
        # Asegurar que los nuevos límites no excedan los originales
        low = np.maximum(low, joint_limits_low)
        high = np.minimum(high, joint_limits_high)
        
        # Asegurar que low <= high
        high = np.maximum(low, high) 
        return low, high

    def sample_target(self, np_random: np.random.Generator, max_reach: float, shoulder_offset_y: float) -> np.ndarray:
        """
        Muestrea una posición objetivo (x,y,z) en el espacio cartesiano.
        La distancia del objetivo aumenta con el nivel del currículo.
        """
        progress = self.get_progress_fraction()
        
        # Interpolar la distancia máxima del objetivo
        # Empieza cerca (min_dist + initial_target_factor * (max_reach - min_dist)) y va hacia max_reach
        current_max_rho = self.min_target_dist + (max_reach - self.min_target_dist) * (self.initial_target_factor + progress * (1.0 - self.initial_target_factor))
        current_min_rho = self.min_target_dist # Siempre al menos min_target_dist
        
        rho = np_random.uniform(current_min_rho, current_max_rho) # Distancia radial
        theta = np_random.uniform(0, np.pi)    # Ángulo polar (desde el eje Z positivo)
        phi = np_random.uniform(0, 2 * np.pi)  # Ángulo azimutal (en el plano XY)
        
        # Coordenadas esféricas a cartesianas (asumiendo origen en el hombro)
        x = rho * np.sin(theta) * np.cos(phi)
        y_robot_frame = rho * np.sin(theta) * np.sin(phi) 
        z = rho * np.cos(theta)
        
        # Ajustar 'y' por el offset del hombro si el target es relativo al centro del robot
        # Si la cinemática directa ya considera el hombro como origen, no se necesita este ajuste aquí.
        # Por ahora, asumimos que (0,0,0) para FK es la base del hombro.
        # El target se da en el frame del hombro.
        # Si el target debe ser en el frame base del robot:
        # y = y_robot_frame - shoulder_offset_y # (si el hombro está en +Y relativo al centro)
        # O y = y_robot_frame + shoulder_offset_y # (si el hombro está en -Y relativo al centro)
        # Basado en `self.params['shoulder_offset_y'] = 0.14974`, parece que el hombro está en y > 0.
        # Si FK está definida desde la base del robot, y `shoulder_offset_y` es parte de la primera D-H transform,
        # entonces el target (x,y,z) que generamos aquí debe estar en ese mismo frame.
        # Asumamos que forward_kinematics espera el target en un frame cuyo origen es la articulación RShoulderPitch.
        # Y `shoulder_offset_y` se usa *dentro* de forward_kinematics para la primera transformación.
        # Entonces (x,y,z) aquí es el target relativo al origen de la cadena cinemática.
        
        # Por lo tanto, no ajustamos por shoulder_offset_y aquí, asumiendo que FK lo maneja
        # o que el target es relativo al "hombro".
        y = y_robot_frame

        return np.array([x, y, z], dtype=np.float32)

    def advance(self):
        if self.level < self.num_curriculum_levels - 1:
            self.level += 1
            print(f"Curriculum advanced to level {self.level}/{self.num_curriculum_levels-1}")

    def set_level(self, level: int):
        self.level = np.clip(level, 0, self.num_curriculum_levels -1)
        print(f"Curriculum set to level {self.level}/{self.num_curriculum_levels-1}")