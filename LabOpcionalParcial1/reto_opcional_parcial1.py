import benchmark_functions as bf
import numpy as np

# func es el objeto creado con la funcion a minimizar
func = bf.StyblinskiTang(n_dimensions=3)
print("Mínimo de la función", func.minimum())

# Dado un punto x0, func permite evaluar la función objetivo en x0
x0 = np.array([5.0, 5.0, 5.0])


def numeric_gradient(xi, f, e=0.000001):

    gradient_result = np.zeros(len(xi))

    for i in range(len(xi)):
        xi_k = np.copy(xi)
        xi_k[i] = xi[i]+e
        gradient_result[i] = (f(xi_k)-f(xi))/e

    return gradient_result


print("Gradiente incial:", numeric_gradient(x0, func))


def gradient_descent(f, numeric_gradient_func, xi, n_max=500):

    alpha = None
    e = None
    min_f = float(f(xi))
    x_min = np.copy(xi)

    for i in range(1, 10):
        alpha = i/100
        for j in range(1, 10):

            xk = np.copy(xi)
            e = j/10
            n = 0
            stop = False

            while not stop:

                num_grad = numeric_gradient_func(xk, f)
                xk1 = xk - (alpha*num_grad)

                if n+1 >= n_max:
                    stop = True
                elif np.linalg.norm(xk1 - xk) <= e:
                    stop = True
                else:
                    xk = np.copy(xk1)
                    n += 1

            if float(f(xk)) < min_f:
                min_f = float(f(xk))
                x_min = np.copy(xk)
                best_alpha = alpha
                best_epsilon = e
                best_n = n

    return best_alpha, best_epsilon, x_min, min_f, best_n


x0_values = [-5, 5]
x0 = np.zeros(3)
results = {}
min_f = None

for i in range(len(x0_values)):
    x0[0] = x0_values[i]
    for j in range(len(x0_values)):
        x0[1] = x0_values[j]
        for k in range(len(x0_values)):
            x0[2] = x0_values[k]

            a, e, x_min, min_f, n = gradient_descent(
                func, numeric_gradient, x0)
            results[min_f] = (a, e, x_min, n, x0)
            print(
                f"\nx_init = {x0}\nalpha = {a}\ne = {e}\nx = {x_min}\nmin_f = {min_f}\nn = {n}")

for key in results:
    if key < min_f:
        min_f = key

a, e, x_min, n, x0 = results[min_f]
print(
    f"\nLA MEJOR APROXIMACIÓN:\nx_init = {x0}\nalpha = {a}\ne = {e}\nx = {x_min}\nmin_f = {min_f}\nn = {n}")
