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

- Filas de entrenamiento: 24.261
- Filas de validacion: 6.066
- MAE goles local: 1.026
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

Validacion cronologica 80/20:

- Partidos evaluados: 6.066

Pick optimizado con estrategia live agresiva actual:

- Puntos totales: 23.603
- Puntos promedio por partido: 3.891
- Resultados exactos: 627, 10.3%
- Acierto de resultado: 60.5%
- Ganador + diferencia: 711
- Empates no exactos acertados: 0
- Ganadores acertados: 2.329
- Fallos: 2.399

Estrategia actual:

- Inflacion estrategica de goles: 1.40
- Maximo de goles totales en picks candidatos: 4
- Multiplicador de probabilidad de empates para seleccion de pick: 1.00

Marcador mas probable:

- Puntos totales: 22.451
- Puntos promedio por partido: 3.701
- Resultados exactos: 657
- Ganador + diferencia: 453
- Empates no exactos acertados: 262
- Ganadores acertados: 2.137
- Fallos: 2.557

Conclusion: para estas reglas, optimizar el valor esperado de puntos y calibrar la seleccion de picks supera a elegir simplemente el marcador mas probable.

Validacion anual tipo produccion, congelada al cierre de 2024:

- Entrenamiento: hasta 2024-12-31.
- Estado de equipos congelado al 2024-12-31 para toda la validacion.
- Validacion: 2025-01-01 a 2025-12-31.
- Partidos evaluados: 1.002.
- Puntos promedio por partido: 3.887.
- Resultados exactos: 104, 10.4%.
- Acierto de resultado: 61.4%.

Validacion especifica para calibrar puntos diarios:

- Entrenamiento: eliminatorias previas al Mundial 2018 desde 2015-01-01 mas Mundial 2018 completo.
- Validacion: eliminatorias mundialistas jugadas en 2021.
- Partidos de entrenamiento: 915.
- Partidos evaluados: 609.
- Mejor estrategia: inflacion de goles 0.90, maximo 4 goles totales candidatos, multiplicador de empates 1.21.
- Puntos promedio por partido: 4.038.
- Resultados exactos: 83, 13.6%.
- Acierto de resultado: 61.4%.

Calibracion live Mundial 2026:

- Partidos finalizados cargados: 26.
- Goles reales promedio: 3.154 por partido.
- La estrategia historica 0.90 habria dado 1 exacto y 3.000 puntos por partido.
- El ajuste live 1.40 habria dado 2 exactos y 3.192 puntos por partido.
- Por eso los picks 2026 usan inflacion 1.40 aunque sea peor en validaciones historicas amplias.

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
PYTHONPATH=src .venv/bin/python examples/evaluate_year_window_picks.py
PYTHONPATH=src .venv/bin/python examples/evaluate_2018_cycle_to_2021_qualifiers.py
```
