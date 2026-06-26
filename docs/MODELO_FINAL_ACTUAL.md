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
6. Aplicar contexto competitivo e inflacion dinamica de goles para picks y simulaciones.
7. Convertir goles esperados ajustados en probabilidades de marcadores.
8. Seleccionar el pick que maximiza puntos esperados bajo reglas calibradas.
9. Exportar predicciones del Mundial 2026.
10. Simular fase de grupos para estimar probabilidades de clasificacion.
11. Simular torneo completo aproximado para estimar campeon, subcampeon y tercer puesto.
12. Estimar goleador desde candidatos editables.

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

Validacion cronologica 80/20 sobre 30.327 partidos:

- Dataset completo: 1993-01-01 a 2026-06-17.
- Entrenamiento: 24.261 partidos, 1993-01-01 a 2019-11-19.
- Validacion: 6.066 partidos, 2019-11-19 a 2026-06-17.

- MAE goles local: 1.026.
- MAE goles visitante: 0.826.
- Puntos promedio por partido, estrategia live agresiva: 3.891.
- Exactos, estrategia live agresiva: 627 de 6.066, 10.3%.
- Acierto de resultado, estrategia live agresiva: 60.5%.
- Puntos promedio por partido, marcador mas probable: 3.701.

Validacion anual tipo produccion, congelada al cierre de 2024:

- Entrenamiento: todos los partidos disponibles hasta 2024-12-31.
- Estado de equipos congelado al 2024-12-31 para todos los partidos de validacion.
- Validacion: partidos de 2025-01-01 a 2025-12-31.
- Partidos de entrenamiento: 28.995.
- Partidos de validacion: 1.002.
- MAE goles local: 1.046.
- MAE goles visitante: 0.886.
- Puntos promedio por partido, estrategia live agresiva: 3.887.
- Exactos, estrategia live agresiva: 104 de 1.002, 10.4%.
- Acierto de resultado, estrategia live agresiva: 61.4%.

Validacion especifica para calibrar puntos diarios:

- Entrenamiento: eliminatorias previas al Mundial 2018 desde 2015-01-01 mas Mundial 2018 completo.
- Partidos de entrenamiento: 915.
- Estado de equipos congelado tras el Mundial 2018.
- Validacion: eliminatorias mundialistas jugadas en 2021.
- Partidos de validacion: 609.
- MAE goles local: 1.117.
- MAE goles visitante: 0.921.
- Mejor estrategia por puntos diarios: inflacion de goles 0.90, maximo 4 goles totales candidatos, multiplicador de empates 1.21.
- Puntos promedio por partido: 4.038.
- Exactos: 83 de 609, 13.6%.
- Acierto de resultado: 61.4%.

Calibracion live Mundial 2026:

- Partidos finalizados cargados: 26.
- Goles reales promedio: 3.154 por partido.
- Goles promedio esperados por el modelo pre-torneo: 2.790.
- Ratio real/modelo: 1.130.
- La estrategia historica 0.90 habria dado 1 exacto y 3.000 puntos por partido en esos 26 partidos.
- El mejor ajuste live del grid fue inflacion de goles 1.40, maximo 4 goles totales candidatos, multiplicador de empates 1.00.
- El ajuste live habria dado 2 exactos y 3.192 puntos por partido en esos 26 partidos.

## Simulacion Final

La simulacion de grupos usa 25.000 corridas con seed fija.

La simulacion completa del torneo usa 500.000 corridas con seed fija.

Benchmark posterior a optimizacion:

- 1.000 simulaciones: aproximadamente 0.61 segundos.
- 5.000 simulaciones: aproximadamente 3.08 segundos.
- 100.000 simulaciones: aproximadamente 1 minuto en esta maquina.

Resultados actuales con 500.000 simulaciones:

- Campeon recomendado: Argentina, 20.081%.
- Subcampeon recomendado: Spain, 9.486% de probabilidad marginal de subcampeon.
- Tercer puesto recomendado: England, 8.917% de probabilidad marginal de tercer puesto.
- Goleador recomendado desde candidatos editables: Harry Kane.

## Estrategia De Picks

La estrategia final actual no elige simplemente el marcador mas probable.

Usa:

- Modo live agresivo calibrado contra partidos reales del Mundial 2026 ya finalizados.
- Inflacion base de goles: 1.40, ajustada dinamicamente por contexto competitivo.
- Maximo de goles totales candidatos: 4.
- Multiplicador de empates: 1.00.
- Optimizacion por puntos esperados segun reglas del juego.
- Picks alternativos por perfil `conservative`, `balanced`, `aggressive` y `desperation`.
- Valor estrategico del pick via `strategic_pick_value`, `estimated_pick_popularity`, `differential_multiplier` y `upside_multiplier`.
- Alternativa de empate visible cuando el empate es competitivo, sin forzarla como pick principal

El CSV de predicciones conserva los goles esperados crudos del modelo en `model_home_expected_goals` y `model_away_expected_goals`. Luego separa `context_*_expected_goals`, `strategy_*_expected_goals` y `market_adjusted_*_expected_goals`. Las columnas `home_expected_goals` y `away_expected_goals` siguen existiendo como alias compatible de la capa final usada para picks y simulaciones.

Esto prioriza reaccionar al ritmo real de goles del torneo y habilita picks mas altos como `2-1`, `1-2`, `3-1`, `3-0`, `1-3` y `0-3` cuando el valor esperado lo justifica.

Detalle de columnas y reglas: `docs/ESTRATEGIA_DINAMICA_DE_PICKS.md`.

## Comandos

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py
PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py
PYTHONPATH=src .venv/bin/python examples/evaluate_baseline_picks.py
PYTHONPATH=src .venv/bin/python examples/evaluate_year_window_picks.py
PYTHONPATH=src .venv/bin/python examples/evaluate_2018_cycle_to_2021_qualifiers.py
PYTHONPATH=src .venv/bin/python scripts/predict_world_cup_fixtures.py
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_groups.py
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_tournament.py
PYTHONPATH=src .venv/bin/python scripts/predict_top_scorer.py
PYTHONPATH=src .venv/bin/python scripts/export_dashboard_data.py
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
- La simulacion completa usa 500.000 corridas por defecto en `scripts/simulate_world_cup_tournament.py`.
- El modulo de goleador depende de `data/raw/top_scorer_candidates.csv`; debe actualizarse cuando salgan planteles, roles y pateadores de penales confirmados.
