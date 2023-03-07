import numpy as np


def descenso_gradiente(x0, eps, alpha, max_iter):
    Q = np.array([[1, 3], [-2, 1]])
    B = np.array([[0.5], [1]])
    C = 5
    x = np.array(x0).reshape(-1, 1)  # convertir x0 en una matriz columna
    values = np.zeros((max_iter, 3))
    f_x = (0.5*x.T @ Q @ x + B.T @ x + C)[0]
    print(f_x)
    # guardar valores iniciales
    values[0, :] = np.array([x[0], x[1], f_x]).flatten()
    for k in range(1, max_iter):
        gradiente = Q @ x + B
        x_new = x - alpha * gradiente
        f_new = 0.5 * x_new.T @ Q @ x_new + B.T @ x_new + C
        # guardar valores de la iteración k
        values[k, :] = np.array([x_new[0], x_new[1], f_new[0]]).flatten()
        if np.linalg.norm(x_new - x) < eps:
            return values[:k+1, :]
        x = x_new
    print("El algoritmo no convergió después de ", max_iter, " iteraciones.")
    return values


x0 = [2, 2]
eps = 0.1
alphas = [0.1, 0.5, 0.8]
max_iter = 100

for alpha in alphas:
    values = descenso_gradiente(x0, eps, alpha, max_iter)
    print("alpha:", alpha)
    print("Número de iteraciones: ", values.shape[0])
    print("Valor mínimo encontrado: ", values[-1, 2])
    print("Coordenadas del mínimo: ", values[-1, :2], '\n')
