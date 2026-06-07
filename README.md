# TD-MT-GVRP con NSGA-II

Este proyecto resuelve instancias del problema TD-MT-GVRP usando una metaheuristica
NSGA-II con busqueda local. El codigo lee archivos AMPL `.dat`, construye soluciones
como cromosomas, las decodifica a rutas factibles y reporta soluciones Pareto para
dos objetivos:

- `F1`: emisiones.
- `F2`: costo/tiempo operacional.

## Estructura

```text
algoritmo/
|-- data/
|   |-- Modelo.dat
|   |-- Modelo(1).dat
|   |-- Modelo_grande.dat
|   `-- modelo_intermedio.dat
|-- results/
|   `-- .gitkeep
|-- src/
|   |-- main.py
|   |-- ampl_dat_parser.py
|   |-- data_model.py
|   |-- chromosome.py
|   |-- decoder.py
|   |-- evaluation.py
|   |-- local_search.py
|   |-- metrics.py
|   |-- nsga2.py
|   `-- reporting.py
|-- tests/
|-- pytest.ini
`-- README.md
```

`results/` se deja vacio en Git. Cada ejecucion crea una subcarpeta nueva con los
reportes generados.

## Requisitos

Usar Python 3.11 o superior. Instalar dependencias:

```bash
pip install matplotlib pytest
```

`matplotlib` es necesario para generar graficos. `pytest` solo es necesario si se
quiere ejecutar la suite de pruebas.

## Como ejecutar

Desde la carpeta `algoritmo/`:

```bash
python src/main.py
```

Por defecto carga:

```text
data/Modelo(1).dat
```

Para usar otra instancia:

```bash
python src/main.py data/Modelo_grande.dat
```

Tambien se puede indicar solo el nombre del archivo si esta dentro de `data/`:

```bash
python src/main.py --data Modelo_grande.dat
python src/main.py --data modelo_intermedio.dat
python src/main.py --data Modelo.dat
```

## Datos de entrada

El parser lee archivos AMPL `.dat` con:

- conjuntos `N`, `P`, `K`, `V`;
- deposito origen `O`;
- capacidad `q`;
- demanda `d`;
- ventanas/periodos `LI`, `LS`;
- pesos sigma `sigma`;
- matrices de emisiones `e`, `ee`;
- matrices de costo `g`, `gg`;
- tiempos `T`, `tt`.

El deposito final o dummy se toma siempre como el ultimo nodo declarado en `N`.
Por ejemplo, si `N` termina en `40`, las rutas terminan en `40`.

Los clientes se infieren desde `d`: todo nodo con demanda positiva es cliente. El
origen y el dummy deben tener demanda `0`.

## Representacion de una solucion

Cada individuo usa tres componentes:

- `perm`: permutacion de clientes.
- `cuts`: cortes acumulados que dividen `perm` entre viajes `(v, k)`.
- `alpha`: tiempo de salida o espera por viaje.

Ejemplo conceptual:

```text
perm = [10, 31, 15, 21, 26, 5]
cuts = [0, 3, 6]
```

produce dos viajes:

```text
viaje 1: [10, 31, 15]
viaje 2: [21, 26, 5]
```

El decoder transforma cada viaje en una ruta:

```text
[O, clientes..., dummy]
```

## Decoder y factibilidad

`src/decoder.py` evalua un individuo sin optimizarlo. Para cada arco:

1. determina el periodo segun el tiempo de salida;
2. suma emisiones `e(i,j,p) + ee(j,p)`;
3. suma costo `g(i,j,p) + gg(j,p)`;
4. avanza el tiempo con `T(i,j,p) + tt(j,p)`;
5. registra variables dispersas `X` y `Y`.

Una solucion se marca infactible si:

- falta algun cliente o hay clientes repetidos;
- un viaje supera capacidad `q`;
- se activa un viaje despues de un viaje vacio del mismo vehiculo;
- un arco sale fuera de todos los periodos;
- el servicio termina fuera del periodo seleccionado;
- el vehiculo termina fuera del horizonte de planificacion.

## Metaheuristica

El algoritmo principal esta en `src/nsga2.py` y se configura desde `src/main.py`.
La ejecucion usa NSGA-II para minimizar simultaneamente `F1` y `F2`.

Componentes principales:

- poblacion inicial factible cuando la capacidad lo permite;
- ordenamiento no dominado;
- distancia de crowding;
- seleccion por torneo binario;
- crossover OX sobre `perm`;
- mutacion swap/insert en `perm`;
- mutacion y reparacion de `cuts`;
- mutacion de `alpha`;
- cache de evaluaciones;
- eliminacion de duplicados en seleccion ambiental;
- parada temprana por estancamiento;
- busqueda local VND opcional.

La busqueda local intenta mejorar individuos aplicando vecindarios sobre la
representacion. El presupuesto de evaluaciones se controla con
`local_search_evaluation_budget`.

## Barrido sigma

`main.py` lee los pesos de `sigma` desde el `.dat`. Para cada fila sigma ejecuta
una corrida con pesos `(beta_f1, beta_f2)`.

Antes del barrido principal hace dos corridas auxiliares:

- una enfocada en minimizar `F1`;
- otra enfocada en minimizar `F2`.

Esas corridas estiman la normalizacion de objetivos usada para seleccionar
soluciones ponderadas y compromisos.

## Salidas

Cada ejecucion crea:

```text
results/run_YYYYMMDD_HHMMSS/
```

Dentro de esa carpeta se guardan:

- `objective_normalization.json`;
- `sigma_sweep_summary.csv`;
- `sigma_pareto_solutions.txt`;
- `sigma_pareto_points.png`;
- una carpeta por cada sigma;
- carpetas `normalization_f1` y `normalization_f2`.

Cada carpeta de corrida contiene:

- `summary.txt`: datos, configuracion, metricas, advertencias y soluciones;
- `pareto_points.txt`: puntos Pareto unicos y frente bruto;
- `generation_history.txt`: evolucion por generacion;
- `execution.log`: progreso de la corrida;
- graficos `.png` de frente Pareto, mejores objetivos, factibilidad e hipervolumen;
- `config.json`: configuracion reproducible.

## Interpretacion rapida

Para revisar si una corrida esta bien, mirar primero:

- `Final feasible count`: debe ser mayor que `0`.
- `Violations`: las soluciones seleccionadas deberian decir `none`.
- `Final Front0 unique objective points`: mientras mas alto, mayor diversidad.
- `Global non-dominated solutions`: soluciones finales realmente no dominadas.
- advertencias automaticas al final de `summary.txt`.

Si todos los individuos son factibles pero hay solo un punto unico por corrida,
el resultado puede ser util como solucion factible, pero la diversidad Pareto es
baja.

## Pruebas

Desde `algoritmo/`:

```bash
pytest
```

Tambien se puede usar:

```bash
python -m unittest discover -s tests
```

## Notas importantes

- `results/` no debe versionar corridas pesadas.
- Para publicar una instancia nueva, agregarla en `data/`.
- Para cambiar parametros del algoritmo, editar `_make_config` en `src/main.py`.
- Para reproducibilidad, revisar `seed`, `population_size`, `generations` y
  `config.json` de cada corrida.
