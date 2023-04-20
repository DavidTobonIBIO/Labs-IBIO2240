import sympy as sp
import numpy as np

x1, x2 = sp.symbols('x1 x2')
f = x1*2 + 2*x2*4 + ((x2*2)/4)
grad = sp.Matrix([sp.diff(f, x1), sp.diff(f, x2)])
hessiana = sp.Matrix([[sp.diff(f, x1, x1), sp.diff(f, x1, x2)], [sp.diff(f, x2, x1), sp.diff(f, x2, x2)]])

print(grad)
print(hessiana)

def invertir_matriz(matriz):
    matriz_np=np.array(matriz)
    inv_matriz = np.linalg.inv(matriz_np)
    return inv_matriz

matriz1 = [[1, 2], [3, 4]]

a=invertir_matriz(matriz1)
print(a)