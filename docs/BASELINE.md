# Baseline Inicial Con Datos Reales

## Fuente

Datos descargados desde `martj42/international_results`:

- `data/raw/results.csv`
- `data/raw/goalscorers.csv`
- `data/raw/shootouts.csv`

## Dataset Procesado

Archivo generado:

```text
data/processed/matches_with_features.csv
```

Resumen de la primera corrida:

- Partidos crudos: 49.400
- Partidos procesados: 30.299

El dataset procesado empieza desde 1993 por defecto y exige al menos 5 partidos previos por seleccion para evitar features sin contexto.

## Features Iniciales

Las variables se calculan antes de cada partido, usando una ventana rolling de partidos anteriores por seleccion:

- `neutral`
- `is_friendly`
- `home_elo`
- `away_elo`
- `elo_diff`
- `elo_diff_scaled`
- `home_elo_expected_result`
- `home_goals_for_avg`
- `home_goals_against_avg`
- `home_points_avg`
- `away_goals_for_avg`
- `away_goals_against_avg`
- `away_points_avg`
- `home_attack_minus_away_defense`
- `away_attack_minus_home_defense`

## Modelo

Baseline inicial:

- Modelo Poisson para goles del local.
- Modelo Poisson para goles del visitante.
- Split cronologico: 80% entrenamiento, 20% validacion.

Resultado de la corrida con Elo pre-partido:

- Filas de entrenamiento: 24.239
- Filas de validacion: 6.060
- MAE goles local: 1.025
- MAE goles visitante: 0.826

Resultado anterior sin Elo:

- MAE goles local: 1.110
- MAE goles visitante: 0.876

El agregado de Elo mejoro el baseline inicial en ambas predicciones de goles.

## Interpretacion

Este baseline ya usa datos reales, pero todavia es simple. Sirve como punto de partida reproducible, no como modelo final.

## Backtest De Puntos Del Juego

Ademas de medir error de goles, se evaluo el rendimiento del modelo bajo las reglas reales del juego.

Archivo generado:

```text
data/processed/baseline_validation_picks.csv
```

Validacion cronologica:

- Partidos evaluados: 6.060

Pick optimizado por reglas del juego:

- Puntos totales: 24.502
- Puntos promedio por partido: 4.043
- Resultados exactos: 838
- Ganador + diferencia: 634
- Empates no exactos acertados: 30
- Ganadores acertados: 2.174
- Fallos: 2.384

Estrategia calibrada:

- Maximo de goles totales en picks candidatos: 2
- Multiplicador de probabilidad de empates para seleccion de pick: 1.23

Marcador mas probable:

- Puntos totales: 23.027
- Puntos promedio por partido: 3.800
- Resultados exactos: 788
- Ganador + diferencia: 436
- Empates no exactos acertados: 214
- Ganadores acertados: 2.075
- Fallos: 2.547

Conclusion: para estas reglas, optimizar el valor esperado de puntos y calibrar la seleccion de picks supera a elegir simplemente el marcador mas probable.

Mejoras recomendadas:

- Agregar ranking FIFA historico pre-partido.
- Separar torneos competitivos de amistosos con pesos distintos.
- Entrenar y validar especificamente contra Mundiales y torneos continentales.
- Agregar valor de plantel y calidad de jugadores cuando haya fuente confiable.
- Agregar cuotas de apuestas para calibracion cercana al kickoff.

## Como Reproducir

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py
PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py
PYTHONPATH=src .venv/bin/python examples/train_baseline_model.py
PYTHONPATH=src .venv/bin/python examples/evaluate_baseline_picks.py
```
