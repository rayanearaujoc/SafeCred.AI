from flask import Flask, render_template
from ModeloEvaluator import ModeloEvaluator

app = Flask(__name__)

@app.route('/')
def index():
    print("🌐 Requisição recebida: /")
    evaluator = ModeloEvaluator("creditcard.csv")
    resultados = evaluator.avaliar_modelos()
    print("✅ Resultados prontos, renderizando página.")
    return render_template("dashboard.html", resultados=resultados.to_dict(orient='records'))

if __name__ == '__main__':
    print("🧠 Iniciando servidor Flask...")
    app.run(debug=True)
