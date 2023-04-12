#tarea_3
import numpy as np
import matplotlib.pyplot as plt

temperatura = np.loadtxt('greenhouse.txt')[:,0]
humedad = np.loadtxt('greenhouse.txt')[:,1]
plt.scatter(humedad, temperatura)
plt.title('Temperatura vs Humedad')
plt.xlabel('Humedad')
plt.ylabel('Temperatura')
plt.show()

