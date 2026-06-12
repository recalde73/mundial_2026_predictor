# Contexto Inicial Del Proyecto

## Objetivo

Construir un sistema de analisis deportivo y ciencia de datos para predecir partidos del Mundial 2026 y maximizar puntos en un juego de predicciones interno del trabajo.

La idea principal no es solamente predecir el resultado mas probable, sino elegir la prediccion que maximice el valor esperado de puntos segun las reglas del juego.

## Reglas Del Juego

### Modo Partido A Partido

- Cada partido se bloquea individualmente antes de su kickoff.
- Las predicciones pueden cambiarse hasta ultimo momento.
- Esto permite actualizar picks con informacion reciente: lesionados, XI confirmado, clima, rotaciones, necesidad de resultado y cuotas de mercado.

### Puntuacion Por Partido

Por cada partido se gana uno solo de estos puntajes. No se acumulan entre si:

- Resultado exacto: 10 puntos.
- Ganador + diferencia de goles: 8 puntos.
- Empate acertado: 6 puntos.
- Ganador acertado: 5 puntos.

En eliminatorias, el bonus de clasificado correcto se suma aparte si el juego lo contempla. Los bonus de campeon, finalista, semifinalista o podio son independientes y se acumulan segun corresponda.

### Resultado Exacto

- Vale 10 puntos.
- Ejemplo: prediccion 2-1 y resultado final 2-1.

### Empate Acertado

- Vale 6 puntos.
- Aplica cuando se predice empate y el partido termina empatado, aunque no sea el mismo marcador.
- Ejemplo: prediccion 1-1 y resultado final 0-0.
- Si el empate exacto coincide, aplica resultado exacto de 10 puntos.

### Ganador Acertado

- Vale 5 puntos.
- Aplica cuando se acierta el equipo ganador, pero no la diferencia de goles.
- Ejemplo: prediccion 1-0 y resultado final 3-1.

### Ganador + Diferencia De Goles

- Vale 8 puntos.
- Aplica cuando se acierta el equipo ganador y la diferencia de goles, pero no el marcador exacto.
- Ejemplo: prediccion 3-1 y resultado final 2-0.

### Goleador Del Torneo

- Vale 10 puntos.
- Se predice el jugador con mas goles del torneo.
- Desempate: cantidad de goles predicha.

### Podio

Prediccion antes del inicio del torneo:

- Campeon correcto: 10 puntos.
- Subcampeon correcto: 8 puntos.
- Tercer puesto correcto: 5 puntos.

## Enfoque Recomendado

Armar el proyecto en cuatro capas:

1. Modelo de fuerza de selecciones.
2. Modelo de goles por partido.
3. Simulador del Mundial completo.
4. Optimizador de predicciones segun la puntuacion del juego.

La capa mas importante para ganar el juego es la cuarta: convertir probabilidades en picks que maximicen puntos esperados.

## Datos A Considerar

### Datos Deportivos Principales

- Ranking FIFA historico.
- Elo internacional.
- Resultados recientes.
- Goles a favor y goles en contra.
- Diferencia de gol.
- xG y xGA si estan disponibles.
- Rendimiento en eliminatorias.
- Rendimiento en torneos grandes.
- Experiencia mundialista.
- Edad promedio del plantel.
- Minutos jugados por titulares en clubes.
- Valor de mercado del plantel.
- Valor de mercado del XI probable.
- Nivel de clubes donde juegan los titulares.
- Lesiones y sanciones.
- Continuidad del director tecnico.
- Forma reciente de jugadores clave.

### Datos Contextuales Del Partido

- Sede.
- Distancia de viaje.
- Dias de descanso.
- Clima.
- Altitud.
- Horario.
- Cercania geografica o localia continental.
- Fase del torneo.
- Necesidad estrategica: ganar, empatar, rotar, cuidar jugadores.

### Datos Socioeconomicos

Usarlos con peso moderado, no como variables dominantes:

- PIB.
- PIB per capita.
- Poblacion.
- Indice de desarrollo humano.
- Inversion deportiva.
- Cantidad de jugadores profesionales exportados.
- Infraestructura futbolistica.
- Valor y competitividad de la liga local.

## Modelado

### Fuerza De Selecciones

Separar la fuerza del equipo en dos componentes:

- Fuerza ofensiva.
- Fuerza defensiva.

Una seleccion fuerte no necesariamente golea. Para elegir marcadores, importa estimar cuantos goles genera y cuantos concede.

### Modelo De Goles

Para cada partido estimar:

- Goles esperados del equipo A.
- Goles esperados del equipo B.
- Probabilidad de cada marcador posible.

Modelos candidatos:

- Poisson simple.
- Poisson bivariado.
- Skellam para diferencia de goles.
- Modelo bayesiano jerarquico.
- XGBoost o LightGBM.
- Ensemble final.

Primera recomendacion: empezar con un modelo explicable basado en Elo, valor del plantel, forma reciente, ataque, defensa y contexto.

### Simulaciones Del Mundial

Simular el torneo completo muchas veces, por ejemplo 100.000 simulaciones.

Guardar probabilidades de:

- Campeon.
- Subcampeon.
- Tercer puesto.
- Semifinalista.
- Finalista.
- Goleador.

### Optimizacion Por Valor Esperado

Para cada marcador candidato calcular puntos esperados.

Ejemplo si se predice 2-1:

```text
EV(2-1) =
10 * P(resultado exacto 2-1)
+ 8 * P(gana el mismo equipo por 1 gol, excepto 2-1)
+ 5 * P(gana el mismo equipo por otra diferencia)
```

Ejemplo si se predice 1-1:

```text
EV(1-1) =
10 * P(resultado exacto 1-1)
+ 6 * P(empate, excepto 1-1)
```

La prediccion recomendada debe ser el marcador con mayor valor esperado, no necesariamente el marcador mas probable.

## Estrategia De Juego

### Fase De Grupos

- Ser mas conservador.
- Favoritos claros: 2-0, 2-1, 1-0 o 3-0 segun diferencia.
- Partidos parejos: evaluar 1-1.
- Equipos defensivos: evitar goleadas.
- Favoritos con alta diferencia de nivel: 2-0, 3-0 o 3-1.

### Eliminatorias

Separar dos conceptos:

- Resultado del partido.
- Equipo clasificado.

Pendiente a confirmar: si el juego toma el resultado a los 90 minutos, a los 120 minutos o despues de penales.

### Podio

No elegir necesariamente los tres equipos con mayor fuerza individual. Elegir la combinacion que maximice valor esperado segun cuadro, cruces y probabilidades simuladas.

### Goleador

Variables importantes:

- Probabilidad de que su seleccion llegue lejos.
- Penales.
- Titularidad.
- Minutos esperados.
- Rivales de grupo.
- Volumen ofensivo del equipo.
- Riesgo de rotacion.
- Historial goleador internacional.

Formula conceptual:

```text
Goles esperados jugador =
goles esperados de su seleccion
* participacion estimada en goles
* minutos esperados
* penales
* probabilidad de avanzar rondas
```

## Roadmap Inicial

1. Confirmar detalles del juego, especialmente eliminatorias.
2. Armar dataset historico de partidos internacionales.
3. Crear ranking propio de selecciones.
4. Crear modelo de goles esperados por equipo.
5. Convertir goles esperados en probabilidades de marcadores.
6. Crear funcion de valor esperado segun reglas del juego.
7. Elegir marcador optimo por partido.
8. Simular el Mundial completo.
9. Calcular probabilidades de campeon, subcampeon, tercero y goleador.
10. Crear dashboard o notebook para actualizar predicciones hasta cada kickoff.

## Preguntas Pendientes

1. En eliminatorias, el resultado predicho corresponde a 90 minutos, 120 minutos o incluye penales?
2. El bonus de clasificado correcto en eliminatorias existe en la version final del juego? Cuantos puntos vale?
3. El fixture del juego se cargara manualmente o desde una fuente externa?
4. El entregable final deseado sera notebook, dashboard web o ambos?
