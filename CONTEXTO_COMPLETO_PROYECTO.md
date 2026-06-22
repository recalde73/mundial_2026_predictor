# Contexto Completo Del Proyecto Mundial 2026 Predictor

## Ubicacion Del Proyecto

```text
/home/ricard/Documents/Proyectos/mundial-2026-predictor
```

## Objetivo General

Construir un sistema de analisis deportivo y ciencia de datos para competir en un juego interno de predicciones del Mundial 2026.

La idea central es maximizar puntos esperados bajo las reglas especificas del juego, no solamente acertar el marcador mas probable.

## Reglas Del Juego

### Partido A Partido

- Cada partido se bloquea individualmente antes del kickoff.
- Se pueden cambiar picks hasta ultimo momento.
- Esto permite actualizar predicciones con resultados previos, lesionados, XI confirmado, clima, cuotas, necesidad de resultado y rotaciones.

### Puntuacion Por Partido

Por cada partido se gana uno solo de estos puntajes:

- Resultado exacto: 10 puntos.
- Ganador + diferencia de goles: 8 puntos.
- Empate acertado: 6 puntos.
- Ganador acertado: 5 puntos.
- Error: 0 puntos.

No se acumulan entre si.

### Bonus Independientes

- Campeon correcto: 10 puntos.
- Subcampeon correcto: 8 puntos.
- Tercer puesto correcto: 5 puntos.
- Goleador del torneo: 10 puntos.

Pendiente a confirmar:

- Si en eliminatorias se predice resultado a 90 minutos, 120 minutos o incluyendo penales.
- Si existe bonus por clasificado correcto y cuantos puntos vale.

## Enfoque Del Modelo

El proyecto se estructuro en capas:

1. Datos reales historicos.
2. Estado y fuerza de selecciones.
3. Modelo de goles esperados.
4. Probabilidades de marcadores.
5. Optimizador de picks segun reglas del juego.
6. Backtesting por puntos reales.
7. Simulacion de fase de grupos.
8. Simulacion completa del torneo.
9. Prediccion de podio.
10. Estimacion de goleador.
11. Dashboard interactivo.

## Fuente De Datos Real

Fuente publica principal:

```text
https://github.com/martj42/international_results
```

Archivos descargados:

```text
data/raw/results.csv
data/raw/goalscorers.csv
data/raw/shootouts.csv
```

`results.csv` incluye resultados historicos y tambien fixtures futuros del Mundial 2026 con goles vacios.

## Datos Manuales Editables

### Resultados Manuales Durante El Torneo

Archivo:

```text
data/manual/match_results_overrides.csv
```

Sirve para cargar resultados reales apenas termina un partido, incluso si la fuente publica todavia no se actualizo.

Formato:

```csv
date,home_team,away_team,home_score,away_score
2026-06-11,Mexico,South Africa,2,1
```

### Candidatos A Goleador

Archivo:

```text
data/raw/top_scorer_candidates.csv
```

Columnas:

```csv
player,team,goal_share,minutes_share,penalty_boost
```

Este archivo debe actualizarse cuando se conozcan planteles, titulares, roles, penales y minutos esperados.

## Modelo Actual

### Features Principales

- Elo historico pre-partido.
- Diferencia Elo.
- Probabilidad esperada Elo.
- Promedio rolling de goles a favor.
- Promedio rolling de goles en contra.
- Promedio rolling de puntos.
- Partido neutral.
- Amistoso vs partido competitivo.

### Modelo De Goles

Se usan dos modelos `PoissonRegressor`:

- Uno para goles del local/equipo A.
- Uno para goles del visitante/equipo B.

### Marcadores

Los goles esperados se convierten en probabilidades de marcadores usando Poisson independiente.

### Optimizador De Picks

El pick recomendado no es necesariamente el marcador mas probable.

Se elige el marcador que maximiza puntos esperados segun reglas del juego.

Estrategia actual live agresiva para Mundial 2026:

- Inflacion estrategica de goles: 1.40.
- Maximo de goles totales candidatos: 4.
- Multiplicador de probabilidad de empates: 1.00.
- Optimizacion por valor esperado de puntos.

## Backtesting

Validacion cronologica 80/20 sobre 6.066 partidos historicos:

```text
MAE goles local: 1.026
MAE goles visitante: 0.826
```

Puntos del juego:

```text
Estrategia live agresiva: 3.891 pts/partido
Marcador mas probable: 3.701 pts/partido
```

Validacion anual entrenando hasta 2024-12-31, congelando estado de equipos al cierre de 2024 y prediciendo 2025:

```text
Estrategia live agresiva: 3.887 pts/partido
Exactos: 10.4%
Acierto de resultado: 61.4%
```

Validacion especifica entrenando con eliminatorias previas al Mundial 2018 mas Mundial 2018, y prediciendo eliminatorias mundialistas 2021:

```text
Estrategia diaria calibrada: 4.038 pts/partido
Exactos: 13.6%
Acierto de resultado: 61.4%
```

Calibracion live Mundial 2026 con 26 partidos finalizados cargados:

```text
Goles reales promedio: 3.154 por partido
Estrategia historica 0.90: 1 exacto, 3.000 pts/partido
Estrategia live 1.40: 2 exactos, 3.192 pts/partido
```

Conclusion: optimizar por reglas del juego supera a elegir el marcador mas probable.

## Simulaciones

### Fase De Grupos

Archivo de salida:

```text
data/processed/world_cup_2026_group_simulation.csv
```

Simula:

- Posicion 1, 2, 3, 4 de grupo.
- Probabilidad de avanzar.

### Torneo Completo

Archivo de salida:

```text
data/processed/world_cup_2026_tournament_simulation.csv
```

Simula:

- Ronda de 32.
- Octavos.
- Cuartos.
- Semifinal.
- Final.
- Campeon.
- Subcampeon.
- Tercer puesto.

La simulacion de grupos usa 25.000 corridas por defecto.

La simulacion completa usa 500.000 corridas por defecto.

El simulador de torneo fue optimizado para correr muchas simulaciones con seed fija y resultados reproducibles.

Limitacion: el bracket de eliminatorias usa una aproximacion por siembra competitiva hasta confirmar el cuadro oficial exacto.

## Resultados Actuales Principales

Con 500.000 simulaciones:

```text
Campeon recomendado: Argentina
Subcampeon recomendado: Spain
Tercer puesto recomendado: England
```

Principales probabilidades de campeon:

```text
Argentina: ~20.1%
Spain: ~15.5%
England: ~12.7%
Brazil: ~10.2%
France: ~8.7%
```

Goleador recomendado actual:

```text
Harry Kane
```

Importante: el modulo de goleador depende del CSV editable de candidatos y debe actualizarse con informacion real de planteles.

## Archivos De Salida Principales

```text
data/processed/world_cup_2026_predictions.csv
data/processed/world_cup_2026_group_simulation.csv
data/processed/world_cup_2026_tournament_simulation.csv
data/processed/world_cup_2026_top_scorer_predictions.csv
```

## Dashboard

Tecnologia:

```text
React + Vite + TypeScript
```

Ubicacion:

```text
dashboard/
```

No usa backend. Consume JSON estatico exportado desde los CSV finales.

JSON del dashboard:

```text
dashboard/public/data/predictions.json
dashboard/public/data/groups.json
dashboard/public/data/tournament.json
dashboard/public/data/top_scorers.json
dashboard/public/data/metadata.json
```

Vistas del dashboard:

- Resumen general.
- Picks partido a partido.
- Simulacion de grupos.
- Simulacion completa del torneo.
- Ranking de goleador.

Comando para abrir:

```bash
cd /home/ricard/Documents/Proyectos/mundial-2026-predictor/dashboard
npm run dev
```

URL habitual:

```text
http://localhost:5173
```

## Comando Unico Del Producto Final

Desde la raiz del proyecto:

```bash
cd /home/ricard/Documents/Proyectos/mundial-2026-predictor
.venv/bin/python scripts/run_full_pipeline.py
```

Este comando ejecuta:

1. Descarga de datos reales.
2. Construccion del dataset.
3. Backtest de picks.
4. Predicciones de fixtures futuros.
5. Simulacion de grupos.
6. Simulacion completa del torneo con 500.000 corridas.
7. Estimacion de goleador.
8. Exportacion de JSON para dashboard.

## Flujo Durante El Mundial

Antes de cada partido:

1. Cargar resultados ya terminados en `data/manual/match_results_overrides.csv`.
2. Ajustar `data/raw/top_scorer_candidates.csv` si hay cambios de lesion, titularidad, penales o minutos.
3. Ejecutar `.venv/bin/python scripts/run_full_pipeline.py`.
4. Abrir dashboard.
5. Revisar picks del partido siguiente.

## Tests Y Verificacion

Suite actual:

```text
42 tests passed
```

Comando:

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

Dashboard verificado con:

```bash
cd dashboard
npm run build
```

## Documentacion Relacionada

```text
CONTEXTO_INICIAL.md
docs/DATOS.md
docs/BASELINE.md
docs/MODELO_FINAL_ACTUAL.md
docs/ROADMAP_STATUS.md
docs/DASHBOARD.md
docs/ACTUALIZACION_DURANTE_TORNEO.md
```

## Limitaciones Conocidas

- Falta confirmar si eliminatorias del juego se predicen a 90 minutos, 120 minutos o con penales.
- Falta confirmar si hay bonus de clasificado correcto y cuantos puntos vale.
- El bracket de eliminatorias es aproximado hasta confirmar el cuadro oficial exacto.
- No integra aun cuotas de apuestas.
- No integra aun XI confirmado automaticamente.
- Lesiones/sanciones se integran por CSV manual, no automaticamente desde una fuente externa.
- No integra valor de plantel ni datos de mercado.
- Goleador depende de candidatos y supuestos manuales.

## Mejoras Futuras Recomendadas

- Fuente automatica de lesiones/sanciones y XI confirmado.
- Integracion de cuotas de apuestas cercanas al kickoff.
- Ajuste por necesidad tactica en fase de grupos.
- Bracket oficial exacto de eliminatorias 2026.
- Optimizacion conjunta de podio, no solo marginal.
- Mejor modelo de goleador con planteles reales y minutos proyectados.
