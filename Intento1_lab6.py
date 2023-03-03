from sympy import *


x1, x2 = symbols('x1 x2')
x = Matrix([x1, x2])
Q = Matrix([[1, 3], [-2, 1]])
B = Matrix([[1/2], [1]])
C = 5

funcion = (x.T*Q*(x)) + (B.T)*x
funcion = funcion[0] + C
verificacion = funcion.subs([(x1, 2), (x2, 2)])


def descenso_de_gradiente_v1(f_x, x, E: float, step_size: float, max_nit: int) -> list:

    x1, x2 = symbols('x1 x2')
    x_k = x
    sol_matrix = []
    i = 0
    converges = False

    gradiente = Matrix([diff(f_x, x1), diff(f_x, x2)])
    grad_x_k = gradiente.subs([(x1, x_k[0]), (x2, x_k[1])])

    while i < max_nit:
        sol_matrix.append([])
        sol_matrix[i].append(x_k[0])
        sol_matrix[i].append(x_k[1])
        sol_matrix[i].append(f_x.subs([(x1, x_k[0]), (x2, x_k[1])]))

        # descenso de gradiente
        grad_x_k = gradiente.subs([(x1, x_k[0]), (x2, x_k[1])])
        x_n = x_k - step_size*grad_x_k
        resta = x_n - x_k
        norma_l2 = sqrt(abs(resta[0])**2 + abs(resta[1])**2)

        if norma_l2 <= E:
            converges = True
            break

        converges = False
        x_k = x_n
        i += 1

    if not converges:
        sol_matrix.append(f"El algoritmo no convergió.")

    return sol_matrix, i


x1 = 2
x2 = 2
x = Matrix([[x1], [x2]])
E = 0.1
step_sizes = [0.1, 0.5, 0.8]
max_nit = 100
sol = None

for step_size in step_sizes:
    sol = descenso_de_gradiente_v1(funcion, x, E, step_size, max_nit)
    print(f"Step size de {step_size}:")
    for i in sol[0]:
        print(i)
    print(f"{sol[1]} iteraciones\n")
