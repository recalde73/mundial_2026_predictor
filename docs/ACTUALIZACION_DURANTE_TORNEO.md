# Actualizacion Durante El Torneo

## Objetivo

Pulir las predicciones a medida que avanza el Mundial usando informacion nueva antes de cada kickoff.

## Resultado De Partidos Ya Jugados

Si la fuente publica todavia no actualizo un resultado, cargarlo manualmente en:

```text
data/manual/match_results_overrides.csv
```

Formato:

```csv
date,home_team,away_team,home_score,away_score
2026-06-11,Mexico,South Africa,2,1
```

Luego ejecutar:

```bash
.venv/bin/python scripts/run_full_pipeline.py
```

Esto recalcula:

- Dataset historico procesado.
- Elo actualizado.
- Estado actual de selecciones.
- Picks de partidos restantes.
- Simulacion de grupos.
- Simulacion completa del torneo.
- Podio.
- Goleador.
- JSON del dashboard.

## Antes De Cada Partido

Checklist recomendado:

- Cargar resultados de partidos ya terminados.
- Cargar lesiones, sanciones, rotaciones o contexto fuerte en `data/manual/match_context_overrides.csv`.
- Revisar `data/raw/top_scorer_candidates.csv` si cambio titularidad, penales o minutos esperados.
- Regenerar pipeline completo.
- Abrir dashboard y revisar el partido siguiente.

## Contexto Reciente Por Partido

Archivo:

```text
data/manual/match_context_overrides.csv
```

Sirve para ajustar partidos futuros sin contaminar el modelo base. Es una capa manual y auditable para eventos recientes:

- Lesion de jugador importante.
- Suspension por roja o acumulacion de amarillas.
- Arquero titular ausente.
- Rotacion fuerte.
- Once confirmado inesperado.
- Necesidad tactica de ganar o cuidar empate.

Formato:

```csv
date,home_team,away_team,home_attack_multiplier,home_defense_multiplier,away_attack_multiplier,away_defense_multiplier,draw_probability_multiplier,confidence,notes
2026-06-22,France,Iraq,0.92,1.00,1.00,1.00,1.00,high,France without key forward
```

Interpretacion:

- `home_attack_multiplier`: ajusta los goles esperados del local.
- `home_defense_multiplier`: ajusta los goles esperados del visitante. Mayor a `1.00` significa peor defensa local.
- `away_attack_multiplier`: ajusta los goles esperados del visitante.
- `away_defense_multiplier`: ajusta los goles esperados del local. Mayor a `1.00` significa peor defensa visitante.
- `draw_probability_multiplier`: ajusta el valor de picks de empate para ese partido.
- `confidence`: `low`, `medium` o `high` segun calidad de la informacion.
- `notes`: explicacion corta para auditar el ajuste en el dashboard.

Guia inicial de magnitudes:

- Baja de titular normal: `0.96` a `0.98` en ataque o `1.02` a `1.04` en defensa.
- Baja ofensiva clave: `0.88` a `0.95` en ataque.
- Arquero titular ausente: `1.05` a `1.12` en defensa.
- Central o mediocentro defensivo clave suspendido: `1.04` a `1.10` en defensa.
- Rotacion fuerte: `0.88` a `0.94` en ataque y/o `1.05` a `1.12` en defensa.

El modelo conserva `model_home_expected_goals` y `model_away_expected_goals` como salida cruda. Luego genera `context_*_expected_goals` con estos ajustes, aplica la estrategia de picks y recalcula probabilidades.

## Recalibracion En Cada Actualizacion

Cada corrida de actualizacion completa recalibra el sistema con la informacion disponible:

- Descarga datos reales.
- Aplica `match_results_overrides.csv`.
- Reconstruye `matches_with_features.csv`.
- Actualiza Elo y forma reciente.
- Reentrena el modelo final con todos los partidos completados disponibles.
- Aplica `match_context_overrides.csv` a fixtures pendientes.
- Regenera picks partido a partido.
- Corre Monte Carlo de grupos y torneo completo.
- Recalcula podio y goleador.

El simulador Monte Carlo ya corre `25.000` fases de grupo y `500.000` torneos completos. Las probabilidades de campeon del dashboard son la frecuencia con la que cada equipo sale campeon en esas simulaciones.

## Mejoras Futuras Recomendadas

- Agregar archivo de cuotas de apuestas para calibrar probabilidades cerca del kickoff.
- Agregar control de tabla de grupo en vivo para ajustar necesidad tactica.
