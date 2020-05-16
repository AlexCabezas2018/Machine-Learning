import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import time
import random


def integra_mc_bucle(fun, a, b, num_puntos=10000):
    """
    Calculates the integral between a and b of a function by using the Monte Carlo method.
    This function does not use the numpy library so it is less eficient than the other version
    """
    tic = time.process_time()

    # We are trying to pick the highest value for the function, given the interval.
    M = -1
    for number in range(num_puntos):
        M = max(M, fun(random.uniform(a, b)))

    # Now, we generate random point inside the rectangle with area A = (b - a) * M
    Ndebajo = 0
    Ntotal = num_puntos
    for number in range(num_puntos):
        # We compare if the random points are under the function
        if random.uniform(0, M) < fun(random.uniform(a, b)):
            Ndebajo += 1

    # Finally, the formula to get an aproximate value
    I = (Ndebajo / Ntotal) * (b - a) * M

    toc = time.process_time()

    return {"time": 1000 * (toc - tic), "value": I}


def integra_mc_vectores(fun, a, b, num_puntos=10000):
    """ Calcula la integral entre a y b de una funcion dada utilizando el método de Monte Carlo,
    haciendo uso de la librería de numpy. Devuelve el tiempo que ha invertido en realizar la operación """

    tic = time.process_time()

    #  We are trying to pick the highest value for the function, given the interval.
    M = np.max(fun(np.random.uniform(a, b, num_puntos)))

    # Now, we generate random point inside the rectangle with area A = (b - a) * M
    ysr = np.random.uniform(0, M, num_puntos)  # y coordinates randomly generated
    ys_from_random_xs = fun(
        np.random.uniform(a, b, num_puntos)
    )  # f(x) where x is randomly generated

    # We compare if the random points are under the function
    ysfr_under_fun = ys_from_random_xs[ysr < ys_from_random_xs]

    Ntotal = num_puntos
    Ndebajo = len(ysfr_under_fun)

    # Finally, the formula to get an aproximate value
    I = (Ndebajo / Ntotal) * (b - a) * M

    toc = time.process_time()

    return {"time": 1000 * (toc - tic), "value": I}


def compara_tiempos():

    num_points = np.linspace(100, 800000, 30)

    times_loop = []
    times_vectors = []
    func = lambda x: x ** 2

    for points in num_points:
        times_loop.append(
            integra_mc_bucle(func, 10, 20, num_puntos=int(points))["time"]
        )
        times_vectors.append(
            integra_mc_vectores(func, 10, 20, num_puntos=int(points))["time"]
        )

    plt.figure()
    plt.scatter(num_points, times_loop, c="red", label="bucle")
    plt.scatter(num_points, times_vectors, c="blue", label="vectores")

    plt.legend()
    plt.savefig("practica-0/time.png")


compara_tiempos()
