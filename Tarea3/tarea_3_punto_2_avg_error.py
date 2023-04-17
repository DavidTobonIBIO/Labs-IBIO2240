# tarea_3
import numpy as np
import matplotlib.pyplot as plt
import pathlib

file_path = pathlib.Path(__file__).parent.resolve().as_posix() + '/'

temperature = np.loadtxt(file_path + 'greenhouse.txt')[:, 0]
humidity = np.loadtxt(file_path + 'greenhouse.txt')[:, 1]
new_temperature = np.loadtxt(file_path + 'datosNuevos.txt')[:, 0]
new_humidity = np.loadtxt(file_path + 'datosNuevos.txt')[:, 1]

# a
plt.scatter(humidity, temperature)
plt.title('Temperatura vs Humedad')
plt.xlabel('Humedad (%)')
plt.ylabel('Temperatura (°C)')
plt.show()

# b
def find_betas_and_H(temperature, humidity):
    T = temperature
    H = np.ones((len(temperature), 2))

    for i in range(len(temperature)):
        H[i, 1] = humidity[i]

    betas = np.linalg.inv(H.T.dot(H)).dot(H.T).dot(T)
    return betas, H


def cuadratic_error(y, betas, H):
    return (np.linalg.norm(y-np.dot(H, betas)))**2


betas, H = find_betas_and_H(temperature, humidity)
error = cuadratic_error(temperature, betas, H)
avg_error_per_datum = error/len(temperature)

print(
    f'Los betas asociados a la ecuacion son: 𝛽0 = {betas[0]} y 𝛽1 = {betas[1]}\nEl error es: {error}\nEl error promedio por dato es: {avg_error_per_datum}')

# grafica de la regresión con mínimos cuadrados
plt.plot(humidity, temperature, 'o')
plt.plot(humidity, betas[0] + betas[1]*humidity, 'r')
plt.title('Temperatura vs Humedad Linealizado')
plt.xlabel('Humedad (%)')
plt.ylabel('Temperatura (°C)')
plt.show()

# c
def anomalo(humedad, temperatura, betas, avg_error):
    t_anomalo = []
    h_anomalo = []
    temp_model = betas[0] + betas[1]*humedad

    for i in range(len(temperatura)):
        error = (abs(temperatura[i]-(temp_model[i])))**2
        if error > avg_error:
            t_anomalo.append(temperatura[i])
            h_anomalo.append(humedad[i])
    return t_anomalo, h_anomalo


t_normal = []
h_normal = []
t_anomalo = anomalo(new_humidity, new_temperature, betas, avg_error_per_datum)

for i in range(len(new_temperature)):
    if (new_temperature[i]) not in (t_anomalo[0]):
        t_normal.append(new_temperature[i])
        h_normal.append(new_humidity[i])


plt.plot(h_normal, t_normal, 'o')
plt.plot(new_humidity, betas[0] + betas[1]*new_humidity, 'r')
plt.plot(t_anomalo[1], t_anomalo[0], 'X', color='black')
plt.title('Temperatura vs Humedad datos anómalos')
plt.xlabel('Humedad (%)')
plt.ylabel('Temperatura (°C)')
plt.legend(['Datos Iniciales', 'Linealización', 'Datos Anómalos'])
plt.show()
