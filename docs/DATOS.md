# Datos Del Proyecto

## Estado Actual

El proyecto ya tiene un pipeline inicial para consumir datos reales externos automaticamente.

Fuente inicial:

- Repositorio: `martj42/international_results`.
- Archivo de resultados: `results.csv`.
- Archivo de goleadores: `goalscorers.csv`.
- Archivo de penales: `shootouts.csv`.
- URL base: `https://github.com/martj42/international_results`.

Estos archivos se descargan con:

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py
```

Luego se construye un dataset entrenable con:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_training_dataset.py
```

El dataset procesado se guarda en:

```text
data/processed/matches_with_features.csv
```

## Que Ya Esta Implementado

El motor matematico:

- Convertir goles esperados en probabilidades de marcadores.
- Calcular puntos esperados segun las reglas del juego.
- Recomendar el marcador que maximiza puntos esperados.

La capa de datos reales:

- Descarga reproducible de resultados historicos.
- Limpieza de partidos.
- Features rolling previas al partido.
- Elo historico pre-partido calculado desde resultados reales.
- Modelo baseline Poisson entrenable para goles local/visitante.

El ejemplo `examples/recommend_match.py` usa goles esperados cargados manualmente:

```text
Argentina 1.72
Dinamarca 0.91
```

Esos numeros son ilustrativos para probar el sistema, no una prediccion basada en datos reales.

El ejemplo `examples/train_baseline_model.py` entrena el primer baseline real desde `matches_with_features.csv`.

## Capa Editable De Ratings

La siguiente capa estima goles esperados desde una tabla editable de ratings por seleccion:

- `rating`: fuerza general del equipo, estilo Elo.
- `attack`: fuerza ofensiva relativa. Cero significa promedio.
- `defense`: fuerza defensiva relativa. Cero significa promedio. Valores positivos indican mejor defensa.

Ejemplo conceptual:

```csv
team,rating,attack,defense
Argentina,1880,0.18,0.14
Dinamarca,1740,0.06,0.10
```

Estos valores se pueden cargar manualmente al principio y despues reemplazar por datos reales.

## Fuentes Recomendadas Para Integrar

Prioridad alta:

- Elo internacional historico.
- Ranking FIFA.
- Resultados recientes de selecciones.
- Goles a favor y en contra.
- Valor de mercado del plantel.
- Lesiones, sanciones y XI probable.

Prioridad media:

- xG y xGA si se consigue una fuente confiable.
- Dias de descanso.
- Sede, clima, viaje y contexto del grupo.
- Cuotas de apuestas para calibracion.

Prioridad baja o complementaria:

- PIB.
- PIB per capita.
- Poblacion.
- Indicadores socioeconomicos.

## Criterio Importante

Los datos socioeconomicos pueden explicar desarrollo estructural, pero no deberian pesar mas que los datos futbolisticos actuales. Para predecir partidos de seleccion, la calidad actual del plantel y el rendimiento reciente suelen ser mas relevantes.
