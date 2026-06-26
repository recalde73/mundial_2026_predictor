# Estrategia Dinamica De Picks

## Objetivo

Reemplazar la inflacion fija de goles como unica capa de estrategia por una capa auditable que combine contexto competitivo, perfil de riesgo y valor diferencial del pick.

El modelo base sigue siendo el mismo: Poisson/Elo/rolling features. La mejora se aplica despues del modelo puro y antes de elegir marcadores.

## Arquitectura

La salida de `data/processed/world_cup_2026_predictions.csv` conserva las columnas existentes y agrega nuevas capas:

- `model_*_expected_goals`: salida pura del modelo entrenado.
- `context_*_expected_goals`: salida tras ajustes contextuales manuales y automaticos.
- `strategy_*_expected_goals`: salida tras inflacion dinamica de goles.
- `market_adjusted_*_expected_goals`: reservado para calibracion por mercado; hoy es igual a la capa estrategica si no hay cuotas cargadas.
- `home_expected_goals` y `away_expected_goals`: alias compatible con el dashboard y simulaciones, apuntan a la capa estrategica final.

## Contexto Competitivo

`build_competitive_context` infiere el estado del grupo con resultados ya jugados y fixtures pendientes. Usa escenarios W/D/L sobre partidos restantes para detectar incentivos antes del partido.

Columnas principales:

- `group`: grupo inferido desde el fixture graph.
- `is_last_group_match`: ambos equipos juegan su ultimo partido de grupo.
- `home_needs_win`, `away_needs_win`: perder deja al equipo sin camino razonable y el empate no asegura clasificacion.
- `home_draw_is_enough`, `away_draw_is_enough`: un empate deja al equipo matematicamente asegurado en top 2 por puntos.
- `home_already_qualified`, `away_already_qualified`: el equipo ya aseguro top 2 antes del partido.
- `home_eliminated`, `away_eliminated`: el equipo no puede alcanzar top 3 por puntos, criterio conservador por mejores terceros.
- `home_goal_difference_pressure`, `away_goal_difference_pressure`: el equipo esta cerca del corte y puede depender de desempates.
- `match_pressure_score`: score 0-1 de presion competitiva del partido.
- `group_scenario_volatility`: score 0-1 de cantidad de combinaciones de clasificacion posibles.

## Inflacion Dinamica

`dynamic_goal_inflation` reemplaza la aplicacion ciega de `goal_inflation` por partido.

Reglas actuales:

- Ambos clasificados: baja hacia `1.05`.
- Empate conveniente: baja hacia `1.10`.
- Ambos eliminados: baja hacia `1.12`.
- Un equipo necesita ganar: sube al menos a `1.32`.
- Ambos necesitan ganar: sube al menos a `1.45`.
- Presion por diferencia de gol o grupo volatil: sube al menos a `1.35`.
- Favorito fuerte por Elo: sube al menos a `1.30`, salvo contexto cerrado.
- Rango final permitido: `0.85` a `1.50`.

La columna `dynamic_goal_inflation_reason` explica las reglas activadas.

## Perfiles De Riesgo

El pipeline genera cuatro picks por partido:

- `conservative_scoreline`: marcador mas probable dentro del resultado 1X2 mas probable.
- `balanced_scoreline`: marcador que maximiza puntos esperados bajo las reglas actuales.
- `aggressive_scoreline`: balance entre puntos esperados, upside y diferencial.
- `desperation_scoreline`: maximiza valor estrategico aun con menor popularidad/mas varianza.

`risk_profile` define que pick se copia a:

- `recommended_scoreline_by_risk`
- `recommended_scoreline`
- `recommended_expected_points`

El perfil por defecto es `aggressive`, alineado con una posicion baja en ranking.

## Valor Estrategico

Para cada pick recomendado se calculan:

- `estimated_pick_popularity`: aproximacion de popularidad usando probabilidad del marcador, popularidad tipica de marcadores comunes y favoritismo 1X2.
- `differential_multiplier`: premio por picks menos obvios.
- `upside_multiplier`: premio por marcadores con mayor potencial diferencial.
- `strategic_pick_value`: `expected_points * differential_multiplier * upside_multiplier`.

Esto no reemplaza el EV: lo complementa cuando el perfil de riesgo exige recuperar posiciones.

## Mercado

Columnas reservadas para cuotas:

- `market_home_win_probability`
- `market_draw_probability`
- `market_away_win_probability`
- `market_over_2_5_probability`
- `market_under_2_5_probability`
- `market_btts_yes_probability`
- `market_btts_no_probability`
- `home_market_edge`
- `draw_market_edge`
- `away_market_edge`
- `market_disagreement_score`
- `market_warning_flag`
- `market_warning_reason`

Hoy quedan vacias si no hay feed/manual CSV de mercado. La siguiente implementacion deberia cargar un CSV manual y mezclar mercado como calibrador, no como reemplazo del modelo.

## Validaciones

Tests agregados:

- Contexto competitivo genera columnas de necesidad y clasificacion.
- Inflacion dinamica baja cuando ambos equipos ya clasificaron.
- Inflacion dinamica sube cuando ambos necesitan ganar.
- El pick recomendado por perfil queda sincronizado con `recommended_scoreline`.

Comando:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_future.py
```
