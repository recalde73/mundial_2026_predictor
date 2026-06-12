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
- Revisar `data/raw/top_scorer_candidates.csv` si cambio titularidad, penales o minutos esperados.
- Regenerar pipeline completo.
- Abrir dashboard y revisar el partido siguiente.

## Mejoras Futuras Recomendadas

- Agregar archivo manual de lesiones y sanciones.
- Agregar archivo manual de XI confirmado.
- Agregar archivo de cuotas de apuestas para calibrar probabilidades cerca del kickoff.
- Agregar control de tabla de grupo en vivo para ajustar necesidad tactica.
