import joblib
import numpy as np
import os
from flask import Flask, request, render_template, make_response

app = Flask(__name__, static_url_path='/static')

# Importando o modelo gerado no notebook
modelo = joblib.load('model\model.joblib')

# Criando rota da home page
@app.route('/')
def index():
    return render_template('index.html')

# Criando a rota que vai retornar o resultado da predição
@app.route('/verificar', methods=['POST'])
def verificar():
    compra = request.form['compra']
    manutencao = request.form['manutencao']
    portas = request.form['portas']
    pessoas = request.form['pessoas']
    bagageiro = request.form['bagageiro']
    seguranca = request.form['seguranca']
    teste = np.array([[compra, manutencao, portas, pessoas, bagageiro, seguranca]])

    print('DADOS DE TESTE')
    print(f'valor da compra: {compra}')
    print(f'valor da manutenção: {manutencao}')
    print(f'quantidade de portas: {portas}')
    print(f'quantidade de pessoas: {pessoas}')
    print(f'Tamanho do bagageiro: {bagageiro}')
    print(f'segurança estimada: {seguranca}')

   
    classe = modelo.predict(teste)[0]
    print(f'Compensa a compra: {classe}')

    return render_template('index.html',classe=str(classe))

if __name__ == "__main__":
    app.run()