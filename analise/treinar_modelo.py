import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Dados fictícios
X = np.array([
    [25, 2, 1, 500],
    [40, 0, 0, 0],
    [30, 1, 2, 200],
    [50, 0, 0, 0],
    [22, 3, 2, 400]
])
y = np.array([1, 0, 1, 0, 1])

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelo
modelo = LogisticRegression()
modelo.fit(X_scaled, y)

# Salvar como dicionário
with open("modelo_inadimplencia.pkl", "wb") as f:
    pickle.dump({"modelo": modelo, "scaler": scaler}, f)

print("✅ Modelo treinado e salvo com sucesso.")

