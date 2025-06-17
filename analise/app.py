from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Carrega modelo e scaler do arquivo .pkl
with open("modelo_inadimplencia.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["modelo"]
    scaler = data["scaler"]

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    if request.method == 'POST':
        try:
            # Coleta os dados do formulário
            dados = [float(request.form[col]) for col in [
                'IDADE_CLIENTE',
                'QT_PC_VENCIDAS',
                'QT_PC_PAGA_ATRASO',
                'VL_MENSALIDADE_ATRASO'
            ]]
            dados_np = np.array(dados).reshape(1, -1)
            
            # Aplica o scaler antes da predição
            X_input = scaler.transform(dados_np)

            # Faz a previsão
            pred = model.predict(X_input)[0]
        except Exception as e:
            pred = f"Erro: {e}"
    return render_template('index.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
