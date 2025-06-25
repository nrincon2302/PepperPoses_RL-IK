# Pepper Inverse Kinematics with Reinforcement Learning
**Nicolás Rincón Sánchez - 202021963**

Este repositorio contiene todo el código, datos y documentación asociados al proyecto de aprendizaje por refuerzo para resolver la cinemática inversa de los brazos del robot Pepper. Se presentan dos entornos (analítico y simulado), un esquema de currículo para graduar la dificultad, y estudios automáticos de sintonización de hiperparámetros con Optuna.

---

## Estructura del proyecto

```
├── docs/                # Documentación e informe en LaTeX y figuras de apoyo (TODO)
└── src/                 # Código fuente y resultados
    ├── environments/    # Definición de entornos Gymnasium (PepperEnv y PepperAnalyticalEnv)
    ├── graphs/          # Gráficas de resultados (PPO / SAC × Left / Right)
    ├── kinematics/      # Utilidades y modelos de cinemática directa
    ├── resultados_Left/ # Carpetas por trial (PPO-analytical-#, SAC-analytical-#) para el brazo izquierdo
    ├── resultados_Right/# Análogo para el brazo derecho
    ├── scripts/         # Scripts de utilidad (e.g. generación de workspace, pruebas)
    └── workspace_cache/ # Caché de puntos muestreados para simulación
```

---

## Objetivos

- **Objetivo general**  
  Desarrollar una solución basada en Aprendizaje por Refuerzo para que el robot Pepper aprenda a resolver su cinemática inversa en tareas de posicionamiento de sus brazos.

- **Objetivos específicos**  
  1. Modelar un entorno realista en Gymnasium que refleje la dinámica y restricciones cinemáticas de Pepper.  
  2. Diseñar una función de recompensa que minimice el error entre la posición actual del efector final y el objetivo.  
  3. Implementar y entrenar políticas con PPO y SAC, ajustando sus hiperparámetros automáticamente con Optuna.  
  4. Evaluar la precisión y la tasa de éxito de las políticas en simulación mediante pruebas sistemáticas.

---

## Metodología

1. **Definición del problema como PDM**  
   Se formuló la cinemática inversa como un proceso de decisión de Markov:  
   - Estado: ángulos articulares y error cartesiano (x, y, z).  
   - Acción: variaciones de ángulo en cada articulación, con límites para garantizar movimientos suaves.  
   - Recompensa: combinación de mejora de distancia, penalización por proximidad, suavidad y éxito al alcanzar umbrales.

2. **Entornos**  
   - **Analítico** (`PepperAnalyticalEnv`): utiliza modelos matemáticos de posición.  
   - **Simulado** (`PepperArmEnv`): emplea qiBullet para simular física y colisiones.

3. **Currículo de dificultad**  
   Se introdujo un “radio de currículo” que define la zona inicial de muestras alrededor del objetivo. Este radio crece escalonadamente al lograr éxitos consecutivos, facilitando un entrenamiento progresivo.

4. **Optimización de hiperparámetros**  
   Con Optuna y su sampler TPE se lanzaron estudios independientes para PPO y SAC en cada brazo. Cada *trial* explora combinaciones de learning rate, batch size, gamma, etc., y se selecciona la mejor política según la recompensa media.

5. **Validación y visualización**  
   - Se generaron gráficos comparativos de métricas (`curriculum/radius`, `rollout/ep_rew_mean`, `rollout/success_rate`) en `src/graphs`.  
   - Se evaluaron 1 000 episodios por política y umbral de precisión (10 cm, 7.5 cm, 5 cm), cuantificando tasa de éxito.

---

## Uso

1. **Instalación**  
   ```bash
   git clone https://github.com/nrincon2302/PepperPoses_RL-IK
   cd src
   pip install -r requirements.txt
   ```

2. **Entrenamiento y HPO**

   ```bash
   python train_pepper.py \
     --env_type analytical \
     --alg PPO \
     --timesteps 1000000 \
     --trials 5 \
     --side Left \
     --n_envs 4 \
     --required_succ 5
   ```

3. **Generar gráficas de TensorBoard**

   ```bash
   python plot_tb_logs.py
   ```

4. **Pruebas de validación**

   ```bash
   python scripts/run_tests.py --model-path modelos/PPO-analytical-3.zip --side Left --thresholds 0.10 0.075 0.05
   ```

---

## Tecnologías Utilizadas

A continuación se listan las referencias de las principales librerías y frameworks utilizados en este proyecto:

- **Stable-Baselines3**  
  Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. & Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine Learning Research, 22(268), 1–8.  

- **Optuna**  
  Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. En Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

- **qiBullet**  
  Busy, M. & Caniot, M. (2019). *qiBullet, a Bullet-based simulator for the Pepper and NAO robots*. arXiv preprint arXiv:1909.00779.

- **Gymnasium**  
  Towers, M., Kwiatkowski, A., Terry, J., Balis, J. U., De Cola, G., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J., Tan, H. & Younis, O. G. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032.
