# Dashboard Web

## Tecnologia

El dashboard esta construido con React + Vite.

No usa backend. Consume JSON estatico generado desde los CSV finales del modelo.

## Flujo De Datos

CSV finales:

```text
data/processed/world_cup_2026_predictions.csv
data/processed/world_cup_2026_group_simulation.csv
data/processed/world_cup_2026_tournament_simulation.csv
data/processed/world_cup_2026_top_scorer_predictions.csv
```

Exportacion a JSON:

```bash
PYTHONPATH=src .venv/bin/python scripts/export_dashboard_data.py
```

Salida:

```text
dashboard/public/data/predictions.json
dashboard/public/data/groups.json
dashboard/public/data/tournament.json
dashboard/public/data/top_scorers.json
dashboard/public/data/metadata.json
```

## Uso

```bash
cd dashboard
npm install
npm run dev
```

Luego abrir la URL que muestre Vite, normalmente:

```text
http://localhost:5173
```

## Vistas

- Resumen: podio recomendado, goleador, candidatos al titulo y picks fuertes.
- Partidos: predicciones fixture a fixture.
- Grupos: probabilidades de posicion y avance.
- Torneo: probabilidades de semifinal, final, campeon y podio.
- Goleador: ranking editable basado en `data/raw/top_scorer_candidates.csv`.
