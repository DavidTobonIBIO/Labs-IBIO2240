import numpy as np
from sympy import *
import pandas as pd
import matplotlib.pyplot as plt
import time

t, y = symbols('t y')
f_prime = y - 6*t
f_sol = 6*t + 6 - 5*exp(t)
t_0 = 0
y_0 = 1

time = 1
h = 0.02

y_n = y_0
t_n = t_0

# Create empty dataframes
df = pd.DataFrame(columns=['Iteracion', 'y_n', 't_n'])
df2 = pd.DataFrame(columns=['Iteracion', 'y_n', 't_n'])
df3 = pd.DataFrame(columns=['Iteracion', 'y_n', 't_n'])


# SOL EXACTA
f_sol = 6*t + 6 - 5*exp(t)

t_values = np.arange(t_0, time+h, h)
y_values = [f_sol.subs({t: i}).evalf() for i in t_values]
df_exact = pd.DataFrame({'t_n': t_values, 'y_n': y_values})
df_exact.to_excel('exacto.xlsx', index=False)

# EULER
start_time_euler = time()
for i in range(0, int(time/h)+1):
    if i == 0:
        y_n = y_0
        t_n = t_0
    else:
        # Evaluar la derivada
        k1 = f_prime.subs({t: t_n, y: y_n}).evalf().simplify()
        # Nuevo valor de y
        y_n = y_n + h*k1
        # Actualizar t
        t_n = t_n + h

    # Agregar valores a dataframe
    df = df.append({'Iteracion': i, 'y_n': round(y_n, 7),
                   't_n': round(t_n, 2)}, ignore_index=True)
    df.to_excel('euler.xlsx', index=False)
end_time_euler = time()
elapsed_time_euler = end_time_euler - start_time_euler
print("Tiempo de ejecución del método de Euler: ",
      elapsed_time_euler, "segundos")


# Heun's method
y_n = y_0
t_n = t_0
for i in range(0, int(time/h)+1):
    if i == 0:
        y_n = y_0
        t_n = t_0
    else:
        # slope at current point
        k1 = f_prime.subs({t: t_n, y: y_n}).evalf().simplify()
        # slope at next point
        k2 = f_prime.subs({t: t_n+h, y: y_n+h*k1}).evalf().simplify()
        y_n = y_n + (h/2)*(k1 + k2)
        t_n = t_n + h
    # Add values to dataframe
    df2 = df2.append({'Iteracion': i, 'y_n': round(y_n, 7),
                     't_n': round(t_n, 2)}, ignore_index=True)
    df2.to_excel('mejorado.xlsx', index=False)


# Runge-Kutta method
y_n = y_0
t_n = t_0
for i in range(0, int(time/h)+1):
    if i == 0:
        y_n = y_0
        t_n = t_0
    else:
        # slope at current point
        k1 = f_prime.subs({t: t_n, y: y_n}).evalf().simplify()
        k2 = f_prime.subs({t: t_n+0.5*h, y: y_n+0.5*h*k1}).evalf().simplify()
        k3 = f_prime.subs({t: t_n+0.5*h, y: y_n+0.5*h*k2}).evalf().simplify()
        k4 = f_prime.subs({t: t_n+h, y: y_n+h*k3}).evalf().simplify()

        y_n = y_n + (h/6)*(k1 + 2*k2+2*k3+k4)
        t_n = t_n + h
    # Add values to dataframe
    df3 = df3.append({'Iteracion': i, 'y_n': round(y_n, 7),
                     't_n': round(t_n, 2)}, ignore_index=True)
    df3.to_excel('runge.xlsx', index=False)

# exact_solution = -18.94528
# plt.scatter(2, exact_solution, label='Solución exacta t=2', color = 'red')
plt.plot(df['t_n'], df['y_n'], label='Euler', color='grey')
plt.plot(df2['t_n'], df2['y_n'], label="Mejorado",
         color='green', linewidth=6, linestyle=':')
plt.plot(df3['t_n'], df3['y_n'], label='Runge-Kutta',
         color='blue', linewidth=2)
plt.plot(df_exact['t_n'], df_exact['y_n'], label='Exacta',
         color='orange', linewidth=2, linestyle=':')

plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparación de métodos numéricos para la EDO')
plt.show()
