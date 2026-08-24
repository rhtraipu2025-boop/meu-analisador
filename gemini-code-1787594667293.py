from collections import Counter
import random
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from scipy.stats import chisquare

app = Flask(__name__)


class AnalisadorEstocastico:

  def __init__(self, universo_min=1, universo_max=60, dezenas_por_sorteio=6):
    self.u_min = universo_min
    self.u_max = universo_max
    self.k = dezenas_por_sorteio
    self.sorteios = []
    self.df = pd.DataFrame()

  def carregar_dados(self, matriz_sorteios):
    self.sorteios = [sorted(s) for s in matriz_sorteios]
    self.df = pd.DataFrame(
        self.sorteios, columns=[f'Pos_{i+1}' for i in range(self.k)]
    )
    self.df['Soma'] = self.df.sum(axis=1)
    self.df['Pares'] = self.df[
        [f'Pos_{i+1}' for i in range(self.k)]
    ].apply(lambda r: sum(1 for x in r if x % 2 == 0), axis=1)
    self.df['Impares'] = self.k - self.df['Pares']
    self.df['Amplitude'] = (
        self.df[f'Pos_{self.k}'] - self.df['Pos_1']
    )

  def relatorio_json(self):
    todos_numeros = [num for s in self.sorteios for num in s]
    total = len(todos_numeros)
    freq = Counter(todos_numeros)
    freq_obs = [freq.get(i, 0) for i in range(self.u_min, self.u_max + 1)]
    freq_esp = [total / float(self.u_max - self.u_min + 1)] * (
        self.u_max - self.u_min + 1
    )
    chi2_stat, p_val = chisquare(freq_obs, f_exp=freq_esp)
    dezenas_frias = sorted(
        list(set(range(self.u_min, self.u_max + 1)) - set(todos_numeros))
    )

    return {
        'total_sorteios': len(self.sorteios),
        'total_dezenas_extraidas': total,
        'soma_media': round(float(self.df['Soma'].mean()), 2),
        'proporcao_pares_pct': round(
            float(self.df['Pares'].sum() / total * 100), 2
        ),
        'proporcao_impares_pct': round(
            float(self.df['Impares'].sum() / total * 100), 2
        ),
        'amplitude_media': round(float(self.df['Amplitude'].mean()), 2),
        'chi2_estatistica': round(float(chi2_stat), 4),
        'p_valor': round(float(p_val), 4),
        'conclusao_estocastica': (
            'ADERENTE_AO_ACASO' if p_val > 0.05 else 'DESVIO_ANOMALO'
        ),
        'dezenas_nao_sorteadas': dezenas_frias,
        'top_5_frequentes': freq.most_common(5),
    }

  def simular_monte_carlo(self, qtd_amostras=3):
    amostras = []
    while len(amostras) < qtd_amostras:
      cand = sorted(random.sample(range(self.u_min, self.u_max + 1), self.k))
      soma = sum(cand)
      pares = sum(1 for x in cand if x % 2 == 0)
      amp = cand[-1] - cand[0]
      if 150 <= soma <= 220 and 2 <= pares <= 4 and amp >= 30:
        amostras.append(cand)
    return amostras


# Banco de dados acumulado
banco_de_dados = [
    [30, 35, 38, 39, 46, 50],
    [2, 11, 22, 30, 51, 54],
    [5, 12, 21, 33, 43, 50],
    [5, 7, 17, 51, 56, 59],
    [8, 28, 30, 37, 39, 60],
    [18, 21, 23, 43, 55, 58],
    [8, 12, 23, 27, 42, 43],
    [20, 28, 32, 35, 40, 54],
    [6, 11, 25, 45, 48, 58],
    [1, 11, 24, 33, 35, 59],
    [2, 10, 11, 25, 51, 56],
    [6, 15, 16, 24, 34, 47],
    [14, 19, 42, 45, 48, 54],
    [23, 29, 33, 42, 43, 57],
    [16, 23, 24, 33, 36, 52],
    [2, 6, 27, 39, 44, 50],
]

sistema = AnalisadorEstocastico()
sistema.carregar_dados(banco_de_dados)


@app.route('/', methods=['GET'])
def home():
  return jsonify(
      {'status': 'API Online', 'mensagem': 'Sistema Estocastico Ativo'}
  )


@app.route('/relatorio', methods=['GET'])
def relatorio():
  return jsonify(sistema.relatorio_json())


@app.route('/simular', methods=['GET'])
def simular():
  qtd = request.args.get('qtd', default=3, type=int)
  qtd = min(max(qtd, 1), 20)
  return jsonify({'amostras_simuladas': sistema.simular_monte_carlo(qtd)})


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)