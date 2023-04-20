import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Leer los datos del archivo txt y guardarlos en un dataframe
dataframe_import = pd.read_csv('datosToy.txt', delim_whitespace=True, header=None)
dataframe_import.columns = ['x', 'y']
dataframe_import['x^2'] = dataframe_import['x']**2
dataframe_import['x^3'] = dataframe_import['x']**3
size = dataframe_import.size


# Imprimir los datos cargados del archivo txt
print('Los datos cargados del archivo txt son:')
print(dataframe_import)
print(f'El tamaño de los datos cargados es:')
print(size)


# Punto 3a

def regresion_lineal(data):
    
    data = dataframe_import
    
    #Creacion de la Matriz H
    
    y_vector=data['y'].to_numpy() #Vector y
    
    h_ones=np.ones(len(data)) #Primera columna de la matriz H
    matrizh =np.column_stack((h_ones, data['x'])) #la matriz H final
    
    #Encontrar los valores Betha
    
    b_est=np.dot(matrizh.T, matrizh)
    inversa=np.linalg.inv(b_est)
    resul=np.dot(inversa, matrizh.T)
    bEstrella=np.dot(resul, y_vector)

    betha1 = bEstrella[0]
    betha2 = bEstrella[1]
    
    return betha1, betha2

b1_reg_lineal, b2_reg_lineal = regresion_lineal(dataframe_import)



print('----------------------------Resultados regrescion lineal---------------------------')

print(f'Los valores de Betha1 y Betha2 son: ')
print(f'Betha1 = {b1_reg_lineal}')
print(f'Betha2 = {b2_reg_lineal}')
print(f'Por lo que la recta de regresion lineal es: y = {b1_reg_lineal} + {b2_reg_lineal}x')
print('----------------------------------------------------------------------------------')

x_values_regLin = np.linspace(-2, 2, 1000)
y_values_regLin = b1_reg_lineal + (b2_reg_lineal*x_values_regLin)
plt.plot(x_values_regLin, y_values_regLin, 'r-', label='Recta de Regresión')
plt.scatter(dataframe_import['x'], dataframe_import['y'], label='Datos')
plt.title('Regresión Lineal')
plt.legend()
plt.show()



#Punto 3b

def regresion_cuadratica(data):
    
    data = dataframe_import
    #Creacion de la Matriz H
    
    y_vector=data['y'].to_numpy() #Vector y
    
    h_ones=np.ones(len(data)) #Primera columna de la matriz H
    matrizh =np.column_stack((h_ones, data['x'], data['x^2'])) #la matriz H final
    
    
    b_est=np.dot(matrizh.T, matrizh)
    inversa=np.linalg.inv(b_est)
    resul=np.dot(inversa, matrizh.T)
    bEstrella=np.dot(resul, y_vector)
    
    beta0 = bEstrella[0]
    beta1 = bEstrella[1]
    beta2 = bEstrella[2]

    return beta0, beta1, beta2


b0_reg_cuadratica, b1_reg_cuadratica, b2_reg_cuadratica = regresion_cuadratica(dataframe_import)

print('----------------------------Resultados regresion cuadratica---------------------------')
print('Los valores de Betha0, Betha1 y Betha2 son: ')
print(f'Betha0 = {b0_reg_cuadratica}')
print(f'Betha1 = {b1_reg_cuadratica}')
print(f'Betha2 = {b2_reg_cuadratica}')
print(f'Por lo que la parabola de regresion cuadratica es: y = {b0_reg_cuadratica} + {b1_reg_cuadratica}x + {b2_reg_cuadratica}x^2')
print('----------------------------------------------------------------------------------')

x_values_regCua = np.linspace(-2, 2, 1000)
y_values_regCua = b0_reg_cuadratica + (b1_reg_cuadratica*x_values_regCua) + (b2_reg_cuadratica*x_values_regCua**2)
plt.plot(x_values_regCua, y_values_regCua, 'r-', label='Parabola generada de la Regresión')
plt.scatter(dataframe_import['x'], dataframe_import['y'], label='Datos')
plt.title('Regresión Cuadrática')
plt.legend()
plt.show()

#Punto 3c

def regresion_cubica(data):
    
    data = dataframe_import
    #Creacion de la Matriz H
    
    y_vector=data['y'].to_numpy() #Vector y
    
    h_ones=np.ones(len(data)) #Primera columna de la matriz H
    matrizh =np.column_stack((h_ones, data['x'], data['x^2'], data['x^3'])) #la matriz H final
    
    
    b_est=np.dot(matrizh.T, matrizh)
    inversa=np.linalg.inv(b_est)
    resul=np.dot(inversa, matrizh.T)
    bEstrella=np.dot(resul, y_vector)
    
    beta0 = bEstrella[0]
    beta1 = bEstrella[1]
    beta2 = bEstrella[2]  
    beta3 = bEstrella[3]  
    
    
    return beta0, beta1, beta2, beta3

b0_reg_cubica, b1_reg_cubica, b2_reg_cubica, b3_reg_cubica = regresion_cubica(dataframe_import)

print('----------------------------Resultados regresion cubica---------------------------')
print('Los valores de Betha0, Betha1, Betha2 y Betha 3 son: ')
print(f'Betha0 = {b0_reg_cubica}')
print(f'Betha1 = {b1_reg_cubica}')
print(f'Betha2 = {b2_reg_cubica}')
print(f'Betha3 = {b3_reg_cubica}')
print(f'Por lo que la parabola de regresion cubica es:')
print(f'y = {b0_reg_cubica} + {b1_reg_cubica}x + {b2_reg_cubica}x^2 + {b3_reg_cubica}x^3') 
print('----------------------------------------------------------------------------------')

x_values_regCubi = np.linspace(-2, 2, 1000)
y_values_regCubi = b0_reg_cubica + (b1_reg_cubica*x_values_regCubi) + (b2_reg_cubica*x_values_regCubi**2) + (b3_reg_cubica*x_values_regCubi**3)
plt.plot(x_values_regCubi, y_values_regCubi, 'r-', label='Curva generada de la Regresión Cubica')
plt.scatter(dataframe_import['x'], dataframe_import['y'], label='Datos')
plt.title('Regresión Cubica')
plt.legend()
plt.show()
