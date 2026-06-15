# Estado Contra Roadmap Inicial

## Completado

- Guardar contexto inicial y reglas del juego.
- Descargar datos reales historicos de partidos internacionales.
- Construir dataset procesado sin fuga de informacion futura.
- Calcular Elo historico pre-partido.
- Crear modelo de goles esperados.
- Convertir goles esperados en probabilidades de marcadores.
- Crear funcion de valor esperado segun reglas del juego.
- Elegir marcador optimo por partido.
- Backtestear puntos reales del juego.
- Generar predicciones para fixtures futuros del Mundial 2026.
- Simular fase de grupos.
- Simular torneo completo aproximado.
- Calcular probabilidades de campeon, subcampeon, tercer puesto, semifinal y final.
- Crear recomendacion de podio.
- Crear estimador de goleador desde candidatos editables.
- Crear comando unico de producto final.
- Optimizar simulador de torneo completo y correr 500.000 simulaciones.

## Resuelto Con Aproximacion

- Eliminatorias: el simulador usa bracket aproximado por siembra competitiva porque el cuadro exacto de cruces debe confirmarse contra la fuente oficial final.
- Goleador: existe modelo y CSV editable, pero necesita actualizar candidatos, minutos, penales y roles cuando salgan planteles reales.
- Podio: se recomienda por probabilidades marginales con equipos distintos; una mejora futura es optimizar combinaciones conjuntas exactas.

## Pendiente Por Definicion Del Juego O Datos Externos

- Confirmar si los resultados de eliminatorias del juego se predicen a 90 minutos, 120 minutos o con penales.
- Confirmar si existe bonus de clasificado correcto y cuantos puntos vale.
- Integrar lesiones, XI confirmado y cuotas de apuestas cerca del kickoff.
- Integrar valor de plantel y datos de jugadores desde una fuente confiable.
- Convertir el flujo en dashboard visual si se decide que hace falta.

## Producto Actual

Comando unico:

```bash
.venv/bin/python scripts/run_full_pipeline.py
```

Salidas principales:

```text
data/processed/world_cup_2026_predictions.csv
data/processed/world_cup_2026_group_simulation.csv
data/processed/world_cup_2026_tournament_simulation.csv
data/processed/world_cup_2026_top_scorer_predictions.csv
```
