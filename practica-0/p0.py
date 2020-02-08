import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import time
import random


def integra_mc_bucle(fun, a, b, num_puntos=10000):
    """ Calcula la integral entre a y b de una funcion por el método de Monte Carlo, sin hacer uso
    de las librerias de numpy. Devuelve el tiempo que ha invertido en realizar la operación
    """
    tic = time.process_time()
    function_points = []

    # Generamos puntos pertenecientes a la funcion y nos quedamos con el mayor
    for number in range(num_puntos):
        x = random.uniform(a, b)
        y = fun(x)
        function_points.append((x, y))

    M = function_points[0][1]
    for point in function_points:
        if point[1] > M:
            M = point[1]

    # Generar puntos aleatorios dentro del cuadrado
    random_points = []
    Ndebajo = 0
    Ntotal = 0
    for number in range(num_puntos):
        x = random.uniform(a, b)
        y = random.uniform(0.0, M)

        y_func = fun(x)
        if y < y_func:
            Ndebajo = Ndebajo + 1

        Ntotal = Ntotal + 1

    I = (Ndebajo / Ntotal) * (b - a) * M

    toc = time.process_time()

    return 1000 * (toc - tic)


def integra_mc_vectores(fun, a, b, num_puntos=100000):
    """ Calcula la integral entre a y b de una funcion dada utilizando el método de Monte Carlo,
    haciendo uso de la librería de numpy. Devuelve el tiempo que ha invertido en realizar la operación """

    tic = time.process_time()

    toc = time.process_time()

    return 1000 * (toc - tic)


def compara_tiempos():

    num_points = np.linspace(100, 500000, 30)

    times_loop = []
    times_vectors = []
    func = lambda x: x ** 2

    for points in num_points:
        times_loop.append(integra_mc_bucle(func, 10, 20, num_puntos=int(points)))
        # times_vectors += [integra_mc_vectores(func, a, b)]

    plt.figure()
    plt.scatter(num_points, times_loop, c="red", label="bucle")
    # plt.scatter(sizes, times_vectors, c="blue", label="vectores")

    plt.legend()
    plt.show()
    # plt.savefig("time.png")


compara_tiempos()

