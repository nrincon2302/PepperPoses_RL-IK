# Configuración del Entrenamiento para IK del Robot Pepper

## Hiperparámetros del Algoritmo PPO

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| n_steps | 2048 | Pasos por actualización |
| batch_size | 64 | Tamaño del batch de entrenamiento |
| gamma | 0.99 | Factor de descuento |
| learning_rate | 3e-4 | Tasa de aprendizaje |
| ent_coef | 0.01 | Coeficiente de entropía |
| vf_coef | 0.5 | Coeficiente de función de valor |
| max_grad_norm | 0.5 | Clipping del gradiente |
| gae_lambda | 0.95 | Factor GAE-Lambda |

## Parámetros del Entorno

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| max_steps | 250 | Pasos máximos por episodio |
| action_max | 0.05 | Cambio máximo de ángulo por paso (rad) |
| goal_threshold | 0.02 | Distancia de éxito al objetivo (m) |
| history_size | 3 | Número de estados históricos almacenados |

## Currículo de Aprendizaje

El currículo se divide en 10 niveles que incrementan gradualmente la dificultad:

### Nivel 1-3: Poses Cercanas y Seguras
- Distancia máxima: 30% del alcance máximo
- Ángulos limitados a 30% del rango total
- Enfoque en posturas naturales y estables

### Nivel 4-6: Poses Intermedias
- Distancia máxima: 60% del alcance máximo
- Ángulos limitados a 60% del rango total
- Introducción de posturas más complejas

### Nivel 7-8: Poses Extendidas
- Distancia máxima: 85% del alcance máximo
- Ángulos limitados a 85% del rango total
- Entrenamiento en posturas desafiantes

### Nivel 9-10: Poses Completas
- Acceso al espacio de trabajo completo
- Rango completo de ángulos articulares
- Énfasis en precisión y eficiencia

### Criterios de Avance
- Se avanza de nivel cada 100,000 timesteps
- Se requiere una tasa de éxito mínima del 70% en el nivel actual

## Monitoreo y Métricas

El entrenamiento registra y guarda:
- Recompensa promedio por episodio
- Tasa de éxito por nivel
- Distancia final al objetivo
- Tiempo de convergencia por nivel
- Uso de límites articulares
- Suavidad del movimiento

Los logs se almacenan en:
- TensorBoard: `training_results/{run_name}/tensorboard_logs/`
- CSV detallado: `training_results/{run_name}/detailed_curriculum_episode_stats.csv`
- CSV agregado: `training_results/{run_name}/aggregated_curriculum_stats.csv`

## Recompensas

| Componente | Peso | Descripción |
|------------|------|-------------|
| Mejora de distancia | 30.0 | (prev_dist - current_dist) |
| Proximidad | 15.0 | 1/(1 + 25*dist²) |
| Penalización por movimiento | -0.15 | Suma cuadrada de acciones |
| Penalización por límites | -0.4 | Por cada límite articular violado |
| Bonus por éxito | 100.0 | Al alcanzar el objetivo |

