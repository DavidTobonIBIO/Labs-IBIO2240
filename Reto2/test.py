from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

c = [10, 100, 1000, 10000]

mat_data = loadmat('data.mat')
print(mat_data)
print(mat_data.keys())
data = mat_data['EEG_O1'][0]
time = np.linspace(0, 10, len(data))

def reg_dmatrix_D2(data, time):

    d_matrix = np.zeros((len(time)-2,len(time)))

    for i in range(len(time)-2):
            d_matrix[i][i] = 1
            d_matrix[i][i+1] = -2
            d_matrix[i][i+2] = 1

    return d_matrix

def x_star(data, c, d):
    identity = np.identity(len(data))
    dt = d.transpose()
    dtd= dt.dot(d)
    m_to_inv = identity+(c*dtd)
    x = np.linalg.inv(m_to_inv)@data
    return x, m_to_inv

D2 = reg_dmatrix_D2(data, time)
print(D2.shape)
print(D2)

x_stars_D2 = []
A_x_D2 = []
    
# Obtener los valores filtrados y la matriz que se invierte para cada c
for i in range(len(c)):
    x_star_x, pre_inv_x = x_star(data, c[i], D2)
    x_stars_D2.append(x_star_x)
    A_x_D2.append(pre_inv_x)

# Graficas de los datos filtrados usando D2 para cada c
plt.figure(figsize=(15, 5))
plt.plot(time, data, label="Datos originales")
for i in range(len(c)):
    plt.plot(time, x_stars_D2[i], label=f"c = {c[i]}")
plt.title(f"Señal regularizada para posición horizontal")
plt.xlabel("Instante de tiempo")
plt.ylabel("Posición en x")
plt.legend()
plt.show()

def reg_dmatrix_D1(data, time):

    d_matrix = np.zeros((len(time)-1,len(time)))

    for i in range(len(time)-1):
            d_matrix[i][i] = 1
            d_matrix[i][i+1] = -1

    return d_matrix

D1 = reg_dmatrix_D1(data, time)
print(D1)

x_stars_D1 = []
A_x_D1 = []

# Obtener los valores filtrados y la matriz que se invierte para cada c
for i in range(len(c)):
    x_star_x, pre_inv_x = x_star(data, c[i], D1)
    x_stars_D1.append(x_star_x)
    A_x_D1.append(pre_inv_x)

# Graficas de los datos filtrados usando D1 para cada c
plt.figure(figsize=(15, 5))
plt.plot(time, data, label="Datos originales")
for i in range(len(c)):
    plt.plot(time, x_stars_D1[i], label=f"c = {c[i]}")
plt.title(f"Señal regularizada para posición horizontal")
plt.xlabel("Instante de tiempo")
plt.ylabel("Posición en x")
plt.legend()
plt.show()

def matrix_condition(A):
    return np.linalg.cond(A, 2)

plt.figure()
conds_x_D1 = []
conds_x_D2 = []

# Calcular condión de la matriz invertible para el proceso con D1 y D2 para cada c
for i in range(len(c)):
    conds_x_D1.append(matrix_condition(A_x_D1[i]))
    conds_x_D2.append(matrix_condition(A_x_D2[i]))
    
# Graficar condiciones de las matrices vs c
plt.figure()
plt.plot(c, conds_x_D1, '-o', label="D1")
plt.plot(c, conds_x_D2, '-o', label="D2")
plt.title("Condición de las matrices de regularización para posición horizontal vs c")
plt.xlabel("c")
plt.ylabel("Condición")
plt.legend()
plt.show()

def build_H(data, n):
    H = np.zeros((len(data), n+1))
    
    for i in range(len(H)):
        for j in range(len(H[0])):
            H[i][j] = data[i]**j
    return H  

def cuadratic_error(original_data, betas, H):
    error = np.linalg.norm(original_data-np.dot(H,betas))**2
    return error


def get_betas(H, y):
    Ht = H.transpose()
    HtH = Ht.dot(H)
    Hty = Ht.dot(y)
    inv_HtH = np.linalg.inv(HtH)
    return inv_HtH.dot(Hty)


errors_x_D1 = {}
errors_x_D2 = {}
for i in range(len(c)):
    errors_x_D1[c[i]] = []
    errors_x_D2[c[i]] = []

errors_dict = {}
hoizontal_errors_list = []
# Probar con 10 grados de polinomios
n_list = list(range(11))

for n in n_list:
    for i in range(len(c)):
        # Calcular matriz H para cada set de datos filtrados con un c dado
        H_x_D1 = build_H(x_stars_D1[i], n)
        H_x_D2 = build_H(x_stars_D2[i], n)

        # Calcular betas óptimos para cada set de datos filtrados con un c dado
        betas_x_D1 = get_betas(H_x_D1, x_stars_D1[i])
        betas_x_D2 = get_betas(H_x_D2, x_stars_D2[i])

        # Calcular errores cuadraticos para cada set de datos filtrados
        error_x_D1 = cuadratic_error(data, betas_x_D1, H_x_D1)
        error_x_D2 = cuadratic_error(data, betas_x_D2, H_x_D2)
        
        # Diccionario para obtener información de los parámetros que resultan en cada valor de error
        errors_dict[f'{error_x_D1}-h'] = f"n = {n}, c = {c[i]}, D = D1"
        errors_dict[f'{error_x_D2}-h'] = f"n = {n}, c = {c[i]}, D = D2"
        
        # Listas que contiene los errores de las regresiones
        hoizontal_errors_list.append(error_x_D1)
        hoizontal_errors_list.append(error_x_D2)
        
        # Listas para graficar error vs n
        errors_x_D1[c[i]].append(error_x_D1)
        errors_x_D2[c[i]].append(error_x_D2)
        
# Encontrar los errores cuadráticos mínimos para los datos de posición horizontal y vertical
min_horizontal_error = min(hoizontal_errors_list)

# Obtener los parámetros que resultan en el error cuadrático mínimo
print(f"El error cuadrático mínimo para la regresión de los datos horizontales es {min_horizontal_error}, obtenido con:\n{errors_dict[str(min_horizontal_error)+'-h']}")

def calculate_regresion(H, betas):
    regresion = np.dot(H, betas)
    return regresion

c_hor = 0
n_hor = 6

H_hor = build_H(x_stars_D2[c_hor], n_hor)
B_hor = get_betas(H_hor, x_stars_D2[c_hor])

horizontal_regresion = calculate_regresion(H_hor, B_hor)

plt.figure(figsize=(15,5))
plt.plot(time, data, label="Datos crudos")
plt.plot(time, x_stars_D2[c_hor], label="Datos filtrados con D2")
plt.plot(time, horizontal_regresion, label="Regresión")
plt.title("Comparación de datos crudos, filtrados y su regresión para la posición horizontal del codo")
plt.xlabel("Instante de tiempo")
plt.ylabel("Posición horizontal del codo")
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(time, data, label="Datos crudos")
plt.plot(time, x_stars_D2[c_hor], label="Datos filtrados con D2")
# plt.plot(time, horizontal_regresion, label="Regresión")
plt.title("Comparación de datos crudos, filtrados y su regresión para la posición horizontal del codo")
plt.xlabel("Instante de tiempo")
plt.ylabel("Posición horizontal del codo")
plt.legend()
plt.xlim([3, 4])
plt.show()