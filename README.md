# Mundial 2026 Predictor

Proyecto de analisis deportivo y ciencia de datos para predecir resultados del Mundial 2026 y maximizar puntos en un juego de predicciones.

## Objetivo

Construir un motor que combine datos deportivos, contexto, simulaciones y reglas de puntuacion para recomendar picks partido a partido.

El foco principal es maximizar valor esperado de puntos, no solo elegir el resultado mas probable.

## Estructura

```text
mundial-2026-predictor/
  CONTEXTO_INICIAL.md        # Reglas, enfoque y decisiones iniciales
  README.md                  # Resumen del proyecto
  requirements.txt           # Dependencias iniciales
  data/
    raw/                     # Datos originales
    processed/               # Datos transformados
  docs/                      # Documentacion adicional
  notebooks/                 # Analisis exploratorio y simulaciones
  src/worldcup_predictor/    # Codigo fuente
  tests/                     # Pruebas
```

## Modulos Implementados

`src/worldcup_predictor/scoring.py` contiene la logica base para:

- Calcular puntos por una prediccion y resultado real.
- Calcular valor esperado de una prediccion a partir de probabilidades de marcadores.
- Elegir la mejor prediccion por valor esperado.

`src/worldcup_predictor/poisson.py` contiene el primer modelo probabilistico:

- Convertir goles esperados en probabilidades de marcadores con Poisson independiente.
- Calcular probabilidades 1X2: gana local, empate, gana visitante.
- Listar marcadores mas probables.

`src/worldcup_predictor/recommender.py` contiene el primer recomendador:

- Ranking de marcadores candidatos por puntos esperados.
- Pick recomendado para maximizar el sistema de puntuacion del juego.
- Alternativas ordenadas por valor esperado.
- Filtro de candidatos plausibles para evitar recomendaciones poco accionables como marcadores demasiado altos.

`src/worldcup_predictor/ratings.py` contiene el primer estimador de goles esperados:

- Carga ratings de equipos desde CSV.
- Usa rating general, ataque y defensa para estimar goles esperados.
- Sirve como capa inicial hasta integrar datos reales externos.

`src/worldcup_predictor/pipeline.py` conecta todo:

- Ratings de equipos.
- Estimacion de goles esperados.
- Recomendacion de marcador por puntos esperados.

`src/worldcup_predictor/data_sources.py`, `datasets.py` y `modeling.py` agregan el primer pipeline con datos reales:

- Descarga de resultados historicos internacionales desde `martj42/international_results`.
- Construccion de features rolling previas a cada partido.
- Entrenamiento de un baseline Poisson para goles esperados.

## Datos

El proyecto puede descargar datos reales automaticamente. El archivo `data/raw/team_ratings_example.csv` sigue existiendo como ejemplo manual, pero el flujo principal ahora usa `data/raw/results.csv` y genera `data/processed/matches_with_features.csv`.

La explicacion completa esta en `docs/DATOS.md`.

El resumen del primer baseline entrenado esta en `docs/BASELINE.md`.

El estado del modelo final actual esta en `docs/MODELO_FINAL_ACTUAL.md`.

La auditoria contra el roadmap inicial esta en `docs/ROADMAP_STATUS.md`.

El flujo para actualizar predicciones durante el torneo esta en `docs/ACTUALIZACION_DURANTE_TORNEO.md`.

## Ejemplo Rapido

```bash
PYTHONPATH=src .venv/bin/python examples/recommend_match.py
```

Ejemplo usando ratings desde CSV:

```bash
PYTHONPATH=src .venv/bin/python examples/recommend_from_ratings.py
```

Descargar datos reales:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py
```

Construir dataset entrenable:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py
```

Entrenar baseline real:

```bash
PYTHONPATH=src .venv/bin/python examples/train_baseline_model.py
```

Evaluar puntos historicos del baseline:

```bash
PYTHONPATH=src .venv/bin/python examples/evaluate_baseline_picks.py
```

Generar predicciones para fixtures futuros del Mundial 2026 incluidos en `results.csv`:

```bash
PYTHONPATH=src .venv/bin/python scripts/predict_world_cup_fixtures.py
```

Salida:

```text
data/processed/world_cup_2026_predictions.csv
```

Simular fase de grupos:

```bash
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_groups.py
```

Salida:

```text
data/processed/world_cup_2026_group_simulation.csv
```

Simular torneo completo aproximado:

```bash
PYTHONPATH=src .venv/bin/python scripts/simulate_world_cup_tournament.py
```

Este script corre `100.000` simulaciones por defecto.

Salida:

```text
data/processed/world_cup_2026_tournament_simulation.csv
```

Estimar goleador del torneo desde candidatos editables:

```bash
PYTHONPATH=src .venv/bin/python scripts/predict_top_scorer.py
```

Salida:

```text
data/processed/world_cup_2026_top_scorer_predictions.csv
```

Ejecutar producto final completo:

```bash
.venv/bin/python scripts/run_full_pipeline.py
```

Actualizar resultados reales del Mundial, recalcular todo y desplegar el dashboard local:

```bash
bash scripts/actualizar_datos_mundial.sh
```

Esto hace automaticamente:

- Descarga `data/raw/results.csv`, `goalscorers.csv` y `shootouts.csv`.
- Consulta partidos finalizados del Mundial en ESPN desde `2026-06-11` hasta hoy.
- Actualiza `data/manual/match_results_overrides.csv`.
- Ejecuta el pipeline completo.
- Compila el dashboard.
- Reinicia `vite preview` en `http://localhost:4173`.

Opciones utiles:

```bash
bash scripts/actualizar_datos_mundial.sh --no-deploy
bash scripts/actualizar_datos_mundial.sh --from-date 2026-06-12 --to-date 2026-06-12
bash scripts/actualizar_datos_mundial.sh --port 5173
```

Exportar datos para dashboard:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_dashboard_data.py
```

Levantar dashboard web:

```bash
cd dashboard
npm install
npm run dev
```

## Proximos Pasos

1. Confirmar reglas pendientes de eliminatorias.
2. Cargar datos historicos de partidos internacionales.
3. Crear ranking propio de selecciones.
4. Implementar modelo de goles esperados.
5. Simular Mundial completo.
6. Construir recomendaciones partido a partido.
