# PepperArm RL Simulation and HPO

## ÍNDICE

1. [Descripción general](#descripción-general)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación y dependencias](#instalación-y-dependencias)
4. [Definición del entorno (pepper\_env.py)](#definición-del-entorno-pepper_envpy)

   * 4.1 [CSpace y robot\_graph](#cspace-y-robot_graph)
   * 4.2 [PepperArmEnv: Espacios, Física y Kinemática](#pepperarmenv-espacios-física-y-kinemática)
   * 4.3 [Curriculum Learning](#curriculum-learning)
   * 4.4 [Manejo de estados, recompensas y terminación](#manejo-de-estados-recompensas-y-terminación)
   * 4.5 [Interacción de consola](#interacción-de-consola)
5. [Proceso de entrenamiento (train\_pepper.py)](#proceso-de-entrenamiento-train_pepperpy)

   * 5.1 [Conceptos: Timesteps, Trials, Samples](#conceptos-timesteps-trials-samples)
   * 5.2 [Fine-grained: Argumentos del script](#fine-grained-argumentos-del-script)
   * 5.3 [HPO con Optuna para PPO y SAC](#hpo-con-optuna-para-ppo-y-sac)
   * 5.4 [Callbacks y métricas custom](#callbacks-y-métricas-custom)
   * 5.5 [Salida y almacenamiento de resultados](#salida-y-almacenamiento-de-resultados)
6. [Instrucciones de uso](#instrucciones-de-uso)
7. [Generación de gráficas y análisis](#generación-de-gráficas-y-análisis)

---

## 1. Descripción general

Este repositorio contiene la implementación de un entorno de simulación para los brazos (izquierdo y derecho) del robot Pepper, así como un pipeline de entrenamiento basado en **Stable-Baselines3** (SB3) y **Optuna** para realizar **Hyperparameter Optimization (HPO)** con los algoritmos **PPO** y **SAC**. Mediante **Curriculum Learning**, el agente aprende progresivamente a alcanzar objetivos en el espacio de trabajo factible, iniciando desde metas cercanas a su posición de arranque y ampliando gradualmente la dificultad.

El objetivo es entrenar en simulación la cinemática inversa de cada brazo de Pepper (5 grados de libertad), sin acceder al robot real. Se emplean:

* **CSpace** para generar puntos muestreados del workspace factible.
* **robot\_graph** para calcular cinemática directa y renderizar la configuración del robot.
* **pepper\_env.py** como entorno Gymnasium personalizado.
* **train\_pepper.py** para entrenar con SB3 y realizar HPO.

---

## 2. Estructura del repositorio

A continuación se muestra la estructura general del repositorio:

```
/ (raíz del proyecto)
├── scripts/
│   ├── CSpace.py                # Generación de workspace sampleado
│   ├── robot_graph.py           # Kinemática directa y visualización de Pepper
│   └── (otros scripts auxiliares)  # si existen
├── pepper_env.py                # Definición del entorno Gymnasium (PepperArmEnv)
├── train_pepper.py              # Script principal de entrenamiento y HPO
├── README.md                    # Este archivo explicativo
├── resultados_calibracion/      # Directorio que se crea tras entrenar
│   ├── PPO-1/                   # Carpeta de resultados de PPO trial #1
│   │   ├── monitor.csv          # Registro de episodios (reward, length, ...)
│   │   ├── tb_logs/             # Logs para TensorBoard (policy_loss, value_loss, etc.)
│   │   ├── eval_logs/           # Logs de evaluación periódica
│   │   ├── curriculum_metrics.csv  # Métricas custom (radio currículo, éxitos, ángulos)
│   │   └── final_model.zip      # Modelo PPO entrenado y guardado
│   ├── PPO-2/ ... PPO-N/        # Repetido para cada trial de PPO
│   ├── SAC-1/                   # Carpeta de resultados de SAC trial #1
│   │   ├── monitor.csv
│   │   ├── tb_logs/
│   │   ├── eval_logs/
│   │   ├── curriculum_metrics.csv
│   │   └── final_model.zip
│   ├── SAC-2/ ... SAC-M/        # Repetido para cada trial de SAC
│   ├── PPO_hpo_results.csv      # CSV resumen de todos los trials de PPO (Optuna)
│   └── SAC_hpo_results.csv      # CSV resumen de todos los trials de SAC (Optuna)
├── requirements.txt             # Lista de dependencias (opcional)
└── .gitignore                   # Archivos/dirs ignorados por Git
```

**Notas**:

* `resultados_calibracion/` se genera automáticamente al ejecutar `train_pepper.py`. No lo incluyas en el control de versiones.
* `scripts/` debe incluir `CSpace.py` y `robot_graph.py` (o similar) con las funciones necesarias.

---

## 3. Instalación y dependencias

Para reproducir el entorno y el entrenamiento, se recomiendan los siguientes pasos:

1. **Crear un entorno virtual**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Instalar dependencias**:

   ```text
   gymnasium
   numpy
   scipy
   pandas
   torch
   stable-baselines3
   optuna
   tqdm
   matplotlib
   ```

   Luego:

   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Definición del entorno (`pepper_env.py`)

Este archivo contiene la clase `PepperArmEnv`, heredada de `gym.Env`, que modela un brazo de Pepper (izquierdo o derecho) con cinemática directa basada en `robot_graph.py` y un esquema de Curriculum Learning.

### 4.1 CSpace y `robot_graph.py`

| Archivo             | Contenido / Funciones principales                                                    |
| ------------------- | ------------------------------------------------------------------------------------ |
| `scripts/CSpace.py` | - `generate_workspace_points(side: str, n_samples: int) -> (np.ndarray, np.ndarray)` |

```
                          - Genera una nube de puntos (x,y,z) alcanzables muestreando discretamente todas las combinaciones de ángulos (5 articulaciones) con `n_samples` por articulación.
```

\| `scripts/robot_graph.py` | - Constantes `LEFT_JOINT_LIMITS` y `RIGHT_JOINT_LIMITS`: diccionarios que mapean cada articulación a sus límites (min, max) en radianes.
\- `calculate_joint_positions(side: str, joint_angles: Dict[str, float]) -> List[np.ndarray]`: devuelve lista de posiciones (x,y,z) de cada eslabón hasta el efector final, usando matrices de rotación y traslación sucesivas (cinemática directa).
\- `plot_robot(left_angles: Dict, right_angles: Dict)`: función que dibuja un diagrama 3D del robot Pepper completo (solo brazo izquierdo o derecho, según se pase `None`). Usa Matplotlib 3D.

**Generación de workspace (CSpace)**:

* Discretiza cada articulación en `n_samples` valores uniformes.
* Para cada tupla de 5 ángulos, calcula posición de efector final y registra `(x,y,z)`.
* Retorna `points` (shape: `(#muestras, 3)`) y, opcionalmente, `angles`.

### 4.2 `PepperArmEnv`: Espacios, Física y Kinemática

| Propiedad / Método | Descripción   |
| ------------------ | ------------- |
| `__init__(...)`    | - Parámetros: |

* `side`: "Left" o "Right".

* `render_mode`: "human" o `None` (para no renderizar).

* `max_steps`: pasos máximos por episodio.

* `n_workspace_samples`: densidad del muestreo (ver CSpace).

* `curriculum_start_frac` e `curriculum_increment_frac`: fracciones de `max_distance` para currículo.

* `required_consecutive_successes`: éxitos seguidos necesarios para subir nivel.

* Configura:

1. `joint_limits`, `joint_keys`: límites articulares (5 DOF).
2. `action_space = Box(low=-0.05, high=+0.05, shape=(5,))`: cada acción es un delta angular (radianes).
3. `observation_space = Box(shape=(11,))`: vector `[θ1..θ5, x_cur, y_cur, z_cur, x_goal, y_goal, z_goal]`.
4. `workspace_points` = nube de puntos desde `generate_workspace_points`.
5. `KDTree` para consultar distancia mínima a workspace (función `is_reachable`).
6. `zero_pos`: posición de efector cuando todos los ángulos son 0.
7. `distances_to_zero`: distancias de cada punto de workspace a `zero_pos`. `max_distance = max(...)`.
8. `curriculum_radius = curriculum_start_frac * max_distance`.
9. `success_consecutive = 0`; `episode_count = 0`.
   \| `_init_episode_state()` | Reinicia variables internas (`current_step`, `joint_angles`, `current_pos`, `target_pos`, `prev_distance`) cada vez que se llama `reset()`.                                                                                                |
   \| `is_reachable(point, tol)` | Usa `KDTree` para verificar si `point ∈ ℝ^3` está a distancia ≤ `tol` de algún `workspace_point` (predeterminado `tol=0.02 m`).                                                                                               |
   \| `reset(seed=None, options=None)` | - Inicializa el episodio:
10. Genera ángulos iniciales uniformes en `[low, high]` para cada articulación.
11. Usa `calculate_joint_positions(...)` para obtener `self.current_pos` (posición del efector).
12. Determina `self.target_pos`:

    * Si `options['target_pos']` existe, primero verifica `is_reachable`. Si no, lanza `ValueError`.
    * Sino, llama a `_sample_target()` (ver más abajo).
13. Calcula `prev_distance = ||target_pos - current_pos||`.
14. Construye y retorna `observation = [θs, current_pos, target_pos]` y `info = {'distance': prev_distance, 'joint_angles': copy(θs)}`.  |
    \| `_sample_target()`      | - Calcula `distances_from_init = ||workspace_points - current_pos||` (posición inicial).

* Filtra los puntos con `distances_from_init ≤ curriculum_radius`.
* Si no quedan, usa todos los `workspace_points`.
* Muestrea uniformemente uno de ellos, le añade ruido uniforme en `[-0.02, +0.02]` en cada coordenada, y retorna esa posición como meta.                                                                                      |
  \| `step(action)`         |

1. `current_step += 1`.
2. Aplica acción: `new_angles = clip(self.joint_angles + action, low, high)`.
3. Calcula `new_pos = calculate_joint_positions(...)[-1]`.
4. `distance = ||target_pos - new_pos||`.
5. `hit_limits = True` si algún ángulo está en (o casi en) su límite.
6. `reward = _compute_reward(distance, prev_distance, action, hit_limits)`.
7. Actualiza `self.joint_angles = new_angles; self.current_pos = new_pos; self.prev_distance = distance`.
8. `terminated = (distance ≤ goal_threshold)`, `truncated = (current_step ≥ max_steps)`.
9. Si `terminated == True`:

   * `self.success_consecutive += 1`.
   * Si `success_consecutive ≥ required_consecutive_successes`: llama a `_update_curriculum()`.
10. Si `terminated == False`: `self.success_consecutive = 0`.
11. Construye `observation = [new_angles, new_pos, target_pos]`.
12. `info = { 'distance': distance, 'joint_angles': copy(new_angles), 'is_success': terminated, 'curriculum_radius': curriculum_radius, 'success_consecutive': success_consecutive, 'target_pos': target_pos }`.
13. Retorna `(observation, reward, terminated, truncated, info)`.          |
    \| `_compute_reward(...)`  | Mezcla de componentes:

* **Mejora**: `(prev_distance − distance) × 30.0`.
* **Proximidad**: `15.0 / (1 + 25 × distance²)`.
* **Smoothness**: `−0.15 × ||action||²`.
* **Penalización por límites**: `−0.4` si `hit_limits == True`.
* **Bonus de éxito**: `+100` si `distance ≤ goal_threshold (0.02 m)`.  |
  \| `_update_curriculum()`   | - `episode_count += 1`.
* `curriculum_radius = min(curriculum_radius + curriculum_increment, max_distance)`.
* `success_consecutive = 0` (reinicia la cuenta).  |
  \| `render()`             | Si `render_mode == 'human'`, obtiene `angles_dict = zip(joint_keys, joint_angles)`, llama a `plot_robot(left_angles=..., right_angles=...)` según `self.side`.  |
  \| `close()`              | Placeholder vacío.                                                                                                                                                                                                               |

### 4.3 Curriculum Learning

En cada “episodio”:

1. Se generan ángulos iniciales aleatorios → se calcula `current_pos`.
2. Se filtran los `workspace_points` por aquellos cuya distancia a `current_pos` ≤ `curriculum_radius`.

   * **`curriculum_radius`**: comienza en `curriculum_start_frac × max_distance`, donde `max_distance = máximo(|| workspace_points - zero_pos ||)`.
3. Se elige un punto del workspace filtrado, se le añade ruido ±0.02 m → `target_pos`.
4. Durante el episodio, si el agente alcanza la meta (distancia ≤ 0.02 m) **5 veces consecutivas** (`required_consecutive_successes = 5`), se incrementa `curriculum_radius` sumando `curriculum_increment_frac × max_distance`.

   * Si un episodio termina sin éxito, se reinicia el contador de éxitos consecutivos.

**Parámetros de currículo**:

| Parámetro                            | Significado                                                                                         |
| ------------------------------------ | --------------------------------------------------------------------------------------------------- |
| `max_distance`                       | Distancia máxima en el workspace desde la pose neutra (ángulos=0).                                  |
| `curriculum_start_frac` (0.2)        | Fracción inicial de `max_distance`. Radio inicial = 0.2 × `max_distance`.                           |
| `curriculum_increment_frac` (0.1)    | Fracción de `max_distance` que se añade al radio cada vez que el agente consigue 5 éxitos seguidos. |
| `required_consecutive_successes` (5) | Número de episodios exitosos en fila requeridos antes de aumentar el radio curricular.              |
| `goal_threshold` (0.02 m)            | Cercanía al objetivo para considerarse éxito (≤ 2 cm).                                              |

### 4.4 Manejo de estados, recompensas y terminación

* **Espacio de acción** (`action_space = Box(-0.05, +0.05, (5,), float32)`): cada componente es un delta angular en radianes para cada articulación.
* **Espacio de observación** (`observation_space = Box(-∞, +∞, (11,), float32)`): vector de 11 valores: `[θ1, θ2, θ3, θ4, θ5, x_cur, y_cur, z_cur, x_goal, y_goal, z_goal]`.
* **Recompensa**: combinación de mejora de distancia, proximidad, penalización por límites, suavidad del movimiento y bonus de éxito.
* **Terminación**:

  * `terminated = True` si `||current_pos - target_pos|| ≤ 0.02`.
  * `truncated = True` si se supera `max_steps` (p. ej. 250).
  * Si `terminated == True` → contar para éxitos consecutivos. Si se alcanzan 5 seguidos → subir currículo.
  * Si `terminated == False` al truncarse → episodio fallido y `success_consecutive = 0`.

### 4.5 Interacción de consola

El bloque final bajo `if __name__ == "__main__":` permite correr:

```bash
python pepper_env.py
```

Y hacer un test rápido:

1. Pregunta al usuario: brazo (`Left`/`Right`), `n_workspace_samples`, `curriculum_start_frac`, `curriculum_increment_frac`, `max_steps`, `required_consecutive_successes`.
2. Crea el entorno con esos parámetros.
3. Ejecuta un episodio con acciones aleatorias y muestra:

   * **Estado inicial**: ángulos, posición actual, posición meta, distancia inicial, radio curricular.
   * En cada paso: distancia actual al objetivo, recompensa, éxito/fracaso, cuenta de éxitos seguidos, radio curricular.
   * Al finalizar: mensaje de éxito o truncamiento.

---

## 5. Proceso de entrenamiento (`train_pepper.py`)

Este script automatiza el entrenamiento con **Stable-Baselines3** y la búsqueda de hiperparámetros con **Optuna** para los algoritmos **PPO** y **SAC**.

### 5.1 Conceptos: Timesteps, Trials, Samples

| Término                   | Descripción                                                                                                                                                                              |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timesteps**             | Cada interacción `env.step(action)` cuenta como 1 timestep. En `.learn(total_timesteps=N)`, el modelo ejecuta N timesteps de simulación.                                                 |
| **Trials**                | Cada prueba individual en la búsqueda de hiperparámetros (Optuna). Un trial evalúa un conjunto de parámetros sugerido y entrena el modelo durante `total_timesteps`.                     |
| **n\_workspace\_samples** | (en el entorno) Número de divisiones por articulación para muestrear el workspace. Por ejemplo, `n_samples=8` produce \~8^5=32 768 configuraciones teóricas antes de filtrar duplicados. |

### 5.2 Fine-grained: Argumentos del script

Al ejecutar:

```bash
python train_pepper.py \
  --timesteps 500000 \
  --trials 10 \
  --side Left \
  --n_samples 8 \
  --start_frac 0.2 \
  --incr_frac 0.1 \
  --max_steps 250 \
  --required_succ 5
```

* `--timesteps 500000`: número total de timesteps por trial (por ejemplo, 500 000).
* `--trials 10`: cantidad de trials de HPO para cada algoritmo (PPO y SAC).
* `--side Left`: envía `side='Left'` a `PepperArmEnv` (brazo izquierdo).
* `--n_samples 8`: envía `n_workspace_samples=8`.
* `--start_frac 0.2`: radio curricular inicial = 20% de la distancia máxima.
* `--incr_frac 0.1`: al lograr 5 éxitos consecutivos, aumenta radio en 10% de la distancia máxima.
* `--max_steps 250`: truncar episodio si no alcanza meta en 250 pasos.
* `--required_succ 5`: número de éxitos consecutivos requeridos para incrementar el radio curricular.

Al iniciar, el script muestra en pantalla los valores de `env_kwargs` y comienza la búsqueda de parámetros para **PPO** primero, luego para **SAC**.

### 5.3 HPO con Optuna para PPO y SAC

| Algoritmo | Hiperparámetros a buscar (Espacio de Búsqueda) |
| --------- | ---------------------------------------------- |
| **PPO**   | - `learning_rate`: LogUniform(1e-5, 1e-3)      |
        - `n_steps`: Categorical([128,256,512,1024])  
        - `gamma`: Uniform(0.95,0.9999)  
        - `gae_lambda`: Uniform(0.8,1.0)  
        - `ent_coef`: LogUniform(1e-8,1e-2)  
        - `clip_range`: Uniform(0.1,0.3)  
        - `vf_coef`: Uniform(0.1,1.0)                                                                                                        |


| **SAC**   | - `learning_rate`: LogUniform(1e-5,1e-3)
\- `buffer_size`: Categorical(\[100000,300000,500000])
\- `batch_size`: Categorical(\[64,128,256])
\- `tau`: Uniform(0.005,0.05)
\- `gamma`: Uniform(0.95,0.9999)
\- `train_freq`: Categorical(\[1,2,4,8])
\- `ent_coef`: LogUniform(1e-8,1e-2)                                                                                                      |

**Flujo interno de cada trial**:

1. Optuna sugiere valores para los hiperparámetros.
2. Se genera una carpeta en `resultados_calibracion/<ALG>-<trial_id>/`.
3. Se configura un logger de SB3 (TensorBoard + CSV) en esa carpeta.
4. Se crean dos entornos (train & eval) con `make_env(seed, log_folder, **env_kwargs)`: cada uno envuelto en `Monitor` que escribe en `monitor.csv`.
5. Se instancia el modelo (`PPO(...)` o `SAC(...)`) con los hiperparámetros sugeridos.
6. Se hace `.learn(total_timesteps=<timesteps>)`, con callbacks:

   * `CurriculumLoggingCallback`: guarda en `curriculum_metrics.csv` datos por episodio.
   * `EvalCallback`: evalúa cada `timesteps/10` pasos en `n_eval_episodes=5`.
   * `TqdmStepCallback`: actualiza barra de progreso en consola.
7. Al terminar, se guarda el modelo final (`final_model.zip`).
8. Se evalúa la política en `eval_env` durante 10 episodios deterministas y se devuelve como valor objetivo (media de la recompensa total) a Optuna.

### 5.4 Callbacks y métricas custom

#### 5.4.1 `CurriculumLoggingCallback`

* Se dispara cada vez que un episodio termina (gracias a `Monitor`).
* Extrae de `info`:

  * `total_reward` (recompensa acumulada)
  * `episode_length`
  * `is_success` (1 o 0)
  * `curriculum_radius` (valor actual)
  * `success_consecutive` (éxitos seguidos al finalizar el episodio)
  * `final_joint_angles`
  * `target_pos`
* Agrega una fila a `curriculum_metrics.csv`. El CSV tiene estas columnas:

  | Columna                 | Descripción                                                       |
  | ----------------------- | ----------------------------------------------------------------- |
  | `episode`               | Número consecutivo de episodio (basado en filas ya escritas)      |
  | `total_reward`          | Suma de recompensas en ese episodio                               |
  | `episode_length`        | Cantidad de timesteps que duró ese episodio                       |
  | `is_success`            | 1 si alcanzó la meta antes de `max_steps`, 0 si truncado o falló  |
  | `curriculum_radius`     | Valor del radio curricular al final del episodio                  |
  | `consecutive_successes` | Conteo de éxitos seguidos tras ese episodio                       |
  | `final_joint_angles`    | Lista de 5 floats que representan los ángulos articulares finales |
  | `target_pos`            | Coordenadas `[x, y, z]` de la meta de ese episodio                |

#### 5.4.2 `EvalCallback`

* Evalúa la política en `eval_env` cada `eval_freq = total_timesteps/10`.
* Corre `n_eval_episodes=5` cada vez, de forma determinista.
* Guarda en `eval_logs/` CSV con recompensa promedio y desviación estándar.
* Opción de guardar el "best\_model" (no obligatorio, pero se conserva el último).

#### 5.4.3 `TqdmStepCallback`

* Actualiza la barra de progreso en consola durante `.learn()`.
* Para PPO: incrementa la barra en cantidad `= n_steps` cada vez que termina un rollout.
* Para SAC: incrementa de a 1 timestep cada vez (aproximación).

### 5.5 Salida y almacenamiento de resultados

Al final de cada trial:

1. `monitor.csv`: registro estándar de Gym (episodio, recompensa, largo, si terminó o truncó).
2. `tb_logs/`: archivos para TensorBoard con métricas internas (policy\_loss, value\_loss, entropy, etc.).
3. `eval_logs/`: evaluación periódica (media y std de reward en 5 episodios).
4. `curriculum_metrics.csv`: métricas custom (ver sección anterior).
5. `final_model.zip`: modelo guardado por SB3.

Adicional:

* `PPO_hpo_results.csv` y `SAC_hpo_results.csv`: DataFrames de Optuna con columnas de trial (número), valores de hiperparámetros, valor objetivo (`mean_reward`). Ejemplo de esquema:

| trial\_number | learning\_rate | n\_steps | gamma | gae\_lambda | ent\_coef | clip\_range | vf\_coef | mean\_reward |
| ------------- | -------------- | -------- | ----- | ----------- | --------- | ----------- | -------- | ------------ |
| 0             | 3.45e-04       | 256      | 0.98  | 0.92        | 0.0005    | 0.15        | 0.60     | 245.2        |
| 1             | ...            | ...      | ...   | ...         | ...       | ...         | ...      | ...          |
| ...           |                |          |       |             |           |             |          |              |

---

## 6. Instrucciones de uso

1. **Configurar y activar el entorno virtual** (opcional).
2. **Instalar dependencias** (`pip install -r requirements.txt`).
3. **Verificar** que `scripts/CSpace.py` y `scripts/robot_graph.py` existan y contengan las funciones:

   * `generate_workspace_points`,
   * `calculate_joint_positions`,
   * `plot_robot`,
   * constantes `LEFT_JOINT_LIMITS` / `RIGHT_JOINT_LIMITS`.
4. **Probar el entorno**:

   ```bash
   python pepper_env.py
   ```

   * Verás prompts para ingresar parámetros y un episodio aleatorio impreso en consola.
5. **Ejecutar entrenamiento y HPO**:

   ```bash
   python train_pepper.py --timesteps 500000 --trials 10 --side Left --n_samples 8 --start_frac 0.2 --incr_frac 0.1 --max_steps 250 --required_succ 5
   ```

   * El script correrá primero 10 trials de PPO, luego 10 trials de SAC.
   * Aparecerán barras de progreso indicando cuántos timesteps faltan para cada trial.
   * Al finalizar cada algoritmo, imprimirá en consola los mejores hiperparámetros y la recompensa media correspondiente.
6. **Explorar resultados**:

   * Revisa `resultados_calibracion/` para ver las carpetas `PPO-1`, `PPO-2`, …, `SAC-1`, etc.
   * Abre `monitor.csv` en cada carpeta para ver recompensa y longitud de cada episodio.
   * Abre TensorBoard apuntando a `tb_logs/` para observar evolución de pérdida y función de valor:

     ```bash
     tensorboard --logdir resultados_calibracion/PPO-1/tb_logs
     ```
   * Analiza `curriculum_metrics.csv` para entender cómo creció el radio curricular, cuántos éxitos seguidos obtuvo el agente y las configuraciones articulares finales.
   * Abre `PPO_hpo_results.csv` y `SAC_hpo_results.csv` para comparar trial por trial y ver cuál fue mejor.

---

## 7. Generación de gráficas y análisis

A partir de los datos guardados, se pueden extraer las siguientes gráficas:

1. **Evolución de la recompensa por episodio**:

   * Usar `monitor.csv` de cada carpeta. Dos columnas relevantes: `r` (recompensa) y `l` (longitud).
   * Graficar `episode_id` en el eje X vs `r` en eje Y.
   * También se puede suavizar con promedio móvil (rolling mean) para ver la tendencia.

2. **Evolución de la función de valor y pérdidas**:

   * Abrir TensorBoard con `tb_logs/` o convertir los logs a CSV.
   * Plotear `value_loss`, `policy_loss`, `entropy_loss` vs `timesteps`.

3. **Gráficas de poses del robot**:

   * Extraer `final_joint_angles` de `curriculum_metrics.csv` cada N episodios (por ejemplo, cada 50).
   * Para cada lista de 5 ángulos, usar `robot_graph.plot_robot(left_angles=..., right_angles=None)` para generar una imagen.
   * Guardar la figura (p. ej. con `plt.savefig(...)`) o mostrar en un notebook.
   * Se pueden superponer varias poses en un mismo gráfico 3D para comparar la trayectoria.

4. **Radio de dificultad sobre workspace (CSpace)**:

   * Cargar `scripts/CSpace.py://generate_workspace_points` con el mismo `n_workspace_samples`.
   * `workspace_points` es un array de `(N,3)`.
   * Para cada valor de `curriculum_radius` encontrado en `curriculum_metrics.csv`, asegurar en qué episodio ocurrió el salto de nivel.
   * Graficar todos los puntos `(x,y,z)` de `workspace_points` en color gris claro.
   * Superponer una esfera de radio = `curriculum_radius` centrada en `current_pos` (o en `zero_pos` si se prefiere).
   * Visualizar cómo crece la esfera a lo largo de los niveles.

5. **Estadísticas de recompensa y pérdidas por etapa del currículo**:

   * Dividir `curriculum_metrics.csv` por rangos de episodios donde `curriculum_radius` era constante.
   * Calcular promedio de `total_reward` y desviación estándar en cada rango.
   * Extraer de `tb_logs` (por episodio o por timesteps) el promedio de `value_loss` o `policy_loss` en esos mismos rangos.
   * Graficar una barra o línea mostrando esos promedios vs nivel de currículo.

6. **Comparación PPO vs SAC**:

   * Con `PPO_hpo_results.csv` y `SAC_hpo_results.csv`, seleccionar los mejores trial (mayor `mean_reward`).
   * Tomar la carpeta `PPO-A` y `SAC-B` correspondientes a esos trials.
   * Repetir los pasos 1 a 5 para cada uno y comparar visualmente en el mismo plot.

7. **Otras métricas posibles**:

   * **Precisió n/Recall/F1**: no aplican directamente, ya que no es aprendizaje supervisado.
   * Sin embargo, se puede definir como “precisión” el porcentaje de pasos en los que el agente decrece la distancia al objetivo (es decir, movimientos útiles). O “ratio de pasos exitosos” en cada episodio.
   * También se puede medir el “número medio de pasos para alcanzar la meta” vs nivel de currículo.
