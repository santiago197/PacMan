#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark de UCS y A* con heuristicas nula, Manhattan y Euclidiana."""

import argparse
import csv
import os
import statistics
import time

import layout
import search
import searchAgents
from pacman import GameState


ALGORITMOS = (
    ("UCS", search.uniformCostSearch, search.nullHeuristic),
    ("A* nula", search.aStarSearch, search.nullHeuristic),
    ("Manhattan", search.aStarSearch, searchAgents.manhattanHeuristic),
    ("Euclidiana", search.aStarSearch, searchAgents.euclideanHeuristic),
)

# Cambia esta coordenada para elegir el punto inicial del problema.
POSICION_INICIAL = (1, 15)


def parse_posicion(valor):
    """Convierte 'x,y' en una coordenada de la cuadricula."""
    try:
        x, y = (int(parte.strip()) for parte in valor.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("la posicion debe tener el formato x,y") from exc
    return x, y


def promedio(valores):
    return statistics.mean(valores) if valores else 0.0


def desviacion_estandar(valores):
    return statistics.stdev(valores) if len(valores) > 1 else 0.0


def ejecutar_una_vez(
    nombre, algoritmo, heuristica, nombre_layout, inicio, objetivo
):
    tablero = layout.getLayout(nombre_layout)
    if tablero is None:
        raise ValueError("No se encontro el layout '{}'".format(nombre_layout))
    for etiqueta, posicion in (("inicial", inicio), ("objetivo", objetivo)):
        if not (0 <= posicion[0] < tablero.width and 0 <= posicion[1] < tablero.height):
            raise ValueError(
                "La posicion {} esta fuera del layout: {}".format(etiqueta, posicion)
            )
        if tablero.isWall(posicion):
            raise ValueError(
                "La posicion {} {} es un muro".format(etiqueta, posicion)
            )

    estado = GameState()
    estado.initialize(tablero, 0)
    problema = searchAgents.PositionSearchProblem(
        estado, start=inicio, goal=objetivo, warn=False
    )

    inicio_cronometro = time.perf_counter()
    acciones = algoritmo(problema) if algoritmo is search.uniformCostSearch else algoritmo(
        problema, heuristica
    )
    tiempo = time.perf_counter() - inicio_cronometro

    return {
        "tiempo_ms": tiempo * 1000,
        "nodos_expandidos": problema._expanded,
        "costo": problema.getCostOfActions(acciones),
        "longitud": len(acciones),
    }


def calcular_resumen(nombre, resultados):
    tiempos = [resultado["tiempo_ms"] for resultado in resultados]
    nodos = [resultado["nodos_expandidos"] for resultado in resultados]
    costos = [resultado["costo"] for resultado in resultados]
    return {
        "algoritmo": nombre,
        "iteraciones": len(resultados),
        "tiempo_promedio_ms": promedio(tiempos),
        "tiempo_desviacion_ms": desviacion_estandar(tiempos),
        "nodos_promedio": promedio(nodos),
        "nodos_desviacion": desviacion_estandar(nodos),
        "costo_promedio": promedio(costos),
        "costo_desviacion": desviacion_estandar(costos),
        "costo": resultados[0]["costo"],
        "longitud": resultados[0]["longitud"],
    }


def guardar_csv(resumenes, ruta, nombre_layout, inicio, objetivo):
    carpeta = os.path.dirname(ruta)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    campos = [
        "layout", "inicio", "objetivo", "algoritmo", "iteraciones",
        "tiempo_promedio_ms", "tiempo_desviacion_ms",
        "nodos_promedio", "nodos_desviacion",
        "costo_promedio", "costo_desviacion", "costo", "longitud",
    ]
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for resumen in resumenes:
            fila = dict(resumen)
            fila["layout"] = nombre_layout
            fila["inicio"] = "{},{}".format(*inicio)
            fila["objetivo"] = "{},{}".format(*objetivo)
            escritor.writerow(fila)


def imprimir_tabla(resumenes):
    encabezado = (
        "Algoritmo       Iter.  Tiempo prom. (ms)  Desv. tiempo (ms)  "
        "Nodos prom.  Desv. nodos  Costo prom.  Desv. costo"
    )
    print("\n" + encabezado)
    print("-" * len(encabezado))
    for resumen in resumenes:
        print(
            "{:<15}{:>5}{:>20.4f}{:>20.4f}{:>13.2f}{:>14.2f}"
            "{:>13.2f}{:>14.2f}".format(
                resumen["algoritmo"],
                resumen["iteraciones"],
                resumen["tiempo_promedio_ms"],
                resumen["tiempo_desviacion_ms"],
                resumen["nodos_promedio"],
                resumen["nodos_desviacion"],
                resumen["costo_promedio"],
                resumen["costo_desviacion"],
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara UCS y A* con heuristicas nula, Manhattan y Euclidiana."
        )
    )
    parser.add_argument("--layout", default="bigMaze", help="Layout a utilizar (por defecto: bigMaze)")
    parser.add_argument(
        "--objetivo",
        type=parse_posicion,
        default=(27, 15),
        help="Posicion objetivo en formato x,y (por defecto: 27,15)",
    )
    parser.add_argument(
        "--iteraciones",
        type=int,
        default=1000,
        help="Numero de ejecuciones por algoritmo (por defecto: 1000)",
    )
    parser.add_argument(
        "--salida",
        default="resultados_benchmark.csv",
        help="CSV de resumen (por defecto: resultados_benchmark.csv)",
    )
    args = parser.parse_args()

    if args.iteraciones < 1:
        parser.error("--iteraciones debe ser mayor o igual que 1")

    resumenes = []
    for nombre, algoritmo, heuristica in ALGORITMOS:
        resultados = [
            ejecutar_una_vez(
                nombre,
                algoritmo,
                heuristica,
                args.layout,
                POSICION_INICIAL,
                args.objetivo,
            )
            for _ in range(args.iteraciones)
        ]
        resumenes.append(calcular_resumen(nombre, resultados))

    imprimir_tabla(resumenes)
    guardar_csv(
        resumenes,
        args.salida,
        args.layout,
        POSICION_INICIAL,
        args.objetivo,
    )
    print("\nResumen guardado en: {}".format(args.salida))


if __name__ == "__main__":
    main()
