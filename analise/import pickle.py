import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 🔢 Dados fictícios para treino
# [idade, renda_mensal, valor_emprestimo, tempo_emprego, pontuacao_credito]
X = np.array([
    [25, 3000, 10000, 2, 600],
    [40, 7000, 5000, 10, 750],
    [30, 4000, 7000, 5, 500],
    [50, 9000, 2000, 20, 800],
    [22, 2000, 12000, 1, 400]
])

# 📌 0 = adimplente, 1 = inadimplente
y = np.array([1, 0, 1, 0, 1])

# 🔄 Escalando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🤖 Treinando o modelo
modelo = LogisticRegression()
modelo.fit(X_scaled, y)

# 💾 Salvando o modelo e o scaler juntos
with open("modelo_inadimplencia.pkl", "wb") as f:
    pickle.dump({"modelo": modelo, "scaler": scaler}, f)

print("✅ Modelo salvo com sucesso!")
