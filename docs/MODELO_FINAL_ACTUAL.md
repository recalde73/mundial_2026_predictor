# Modelo Final Actual

## Estado

El proyecto ya genera predicciones reales para fixtures futuros del Mundial 2026 incluidos en la fuente publica `martj42/international_results`.

Archivos principales de salida:

```text
data/processed/world_cup_2026_predictions.csv
data/processed/world_cup_2026_group_simulation.csv
data/processed/world_cup_2026_tournament_simulation.csv
data/processed/world_cup_2026_top_scorer_predictions.csv
```

## Fuente De Datos

- Repositorio: `https://github.com/martj42/international_results`
- Resultados historicos y fixtures: `data/raw/results.csv`
- Goleadores historicos: `data/raw/goalscorers.csv`
- Penales historicos: `data/raw/shootouts.csv`

## Pipeline

1. Descargar datos reales.
2. Procesar partidos historicos completados.
3. Construir features previas a cada partido.
4. Calcular Elo historico pre-partido.
5. Entrenar modelo Poisson de goles esperados.
6. Convertir goles esperados en probabilidades de marcadores.
7. Seleccionar el pick que maximiza puntos esperados bajo reglas calibradas.
8. Exportar predicciones del Mundial 2026.
9. Simular fase de grupos para estimar probabilidades de clasificacion.
10. Simular torneo completo aproximado para estimar campeon, subcampeon y tercer puesto.
11. Estimar goleador desde candidatos editables.

## Modelo

- Modelo de goles local: `PoissonRegressor`.
- Modelo de goles visitante: `PoissonRegressor`.
- Entrenamiento final: todos los partidos historicos procesados disponibles.
- Validacion historica: split cronologico 80/20.

## Features Principales

- Elo local y visitante.
- Diferencia Elo.
- Probabilidad Elo esperada del local.
- Promedio rolling de goles a favor.
- Promedio rolling de goles en contra.
- Promedio rolling de puntos.
- Partido neutral.
- Amistoso vs partido competitivo.

## Backtest

Validacion cronologica sobre 6.060 partidos:

- MAE goles local: 1.025
- MAE goles visitante: 0.826
- Puntos promedio por partido, estrategia calibrada: 4.043
- Puntos promedio por partido, marcador mas probable: 3.800

## Simulacion Final

La simulacion completa del torneo usa 100.000 corridas.

Benchmark posterior a optimizacion:

- 1.000 simulaciones: aproximadamente 0.61 segundos.
- 5.000 simulaciones: aproximadamente 3.08 segundos.
- 100.000 simulaciones: aproximadamente 1 minuto en esta maquina.

Resultados actuales con 100.000 simulaciones:

- Campeon recomendado: Spain, 22.958%.
- Subcampeon recomendado: Argentina, 8.906% de probabilidad marginal de subcampeon.
- Tercer puesto recomendado: England, 6.947% de probabilidad marginal de tercer puesto.
- Goleador recomendado desde candidatos editables: Harry Kane.

## Estrategia De Picks

La estrategia final actual no elige simplemente el marcador mas probable.

Usa:

- Maximo de goles totales candidatos: 2
- Multiplicador de empates: 1.23
- Optimizacion por puntos esperados segun reglas del juego

Esto favorece marcadores conservadores como `1-0`, `0-1`, `2-0`, `0-2` y algunos empates cuando el valor historico lo justifica.

## Comandos

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py
PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py
PYTHONPATH=src .venv/bin/python examples/evaluate_baseline_picks.py
PYTHONPATH=src .venv/bin/python scripts/predict_world_cup_fixtures.py
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_groups.py
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_tournament.py
PYTHONPATH=src .venv/bin/python scripts/predict_top_scorer.py
```

Comando unico:

```bash
.venv/bin/python scripts/run_full_pipeline.py
```

## Limitaciones Actuales

- Todavia no integra valor de plantel ni lesionados.
- Todavia no integra cuotas de apuestas.
- Todavia no ajusta por XI confirmado.
- Los fixtures futuros dependen de que la fuente publica los tenga actualizados.
- El simulador de eliminatorias usa un bracket aproximado por siembra competitiva hasta confirmar el cuadro oficial exacto.
- La simulacion completa usa 100.000 corridas por defecto en `scripts/simulate_world_cup_tournament.py`.
- El modulo de goleador depende de `data/raw/top_scorer_candidates.csv`; debe actualizarse cuando salgan planteles, roles y pateadores de penales confirmados.
