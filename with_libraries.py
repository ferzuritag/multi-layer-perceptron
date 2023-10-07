import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Datos de entrada XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Resultados esperados (salida XOR)
y = np.array([0, 1, 1, 0])

# Crear el modelo
model = MLPClassifier(hidden_layer_sizes=(3,), max_iter=10000, random_state=42)

# Listas para almacenar el error cuadrático medio en cada iteración
mse_values = []

# Entrenar el modelo y almacenar el MSE en cada iteración
for i in range(1, 1001):  # 1000 iteraciones
    model.partial_fit(X, y, classes=[0, 1])
    y_pred = model.predict(X)
    mse = np.mean((y_pred - y) ** 2)
    mse_values.append(mse)

# Graficar el error cuadrático medio a lo largo de las iteraciones
plt.figure(figsize=(8, 6))
plt.plot(range(1, 1001), mse_values)
plt.title('Error Cuadrático Medio (MSE) a lo largo de las Iteraciones')
plt.xlabel('Iteraciones')
plt.ylabel('MSE')
plt.show()