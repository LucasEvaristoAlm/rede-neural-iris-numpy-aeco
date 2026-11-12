# rede-neural-iris-numpy-aeco

Rede neural com uma camada escondida (ReLU + Softmax) implementada do zero com NumPy para classificar o dataset Iris. Inclui backpropagation e gradiente descendente.

## Sobre

Implementação didática, em um único arquivo (`rede_neural_iris.py`), de uma rede neural artificial com uma camada escondida para classificar as três classes do dataset Iris (setosa, versicolor, virginica).

**Arquitetura:**

```
Entrada (4 features) → Camada escondida (8 neurônios, ReLU) → Camada de saída (3 classes, Softmax)
```

**O que é implementado manualmente (somente NumPy):**

- Funções de ativação: `relu` / `relu_derivada`, `softmax` (numericamente estável)
- Função de perda: `cross_entropy` (entropia cruzada categórica com clipping)
- Forward pass e cache para o backpropagation
- Backpropagation com a regra da cadeia (softmax + cross-entropy em forma fechada)
- Gradiente descendente em lote (`atualizar_parametros`)
- Laço de treinamento (`treinar`) com impressão da perda a cada 100 épocas

**Uso do Scikit-Learn (apenas carga e preparação dos dados):**

- `load_iris` — carregar o dataset
- `StandardScaler` — padronização das features (ajustada só no treino, sem vazamento de dados)
- `train_test_split` — divisão treino (80%) / teste (20%) com `stratify`

Todo o restante (One-Hot Encoding, forward, perda, gradientes, métricas) é feito com NumPy puro — sem TensorFlow ou PyTorch.

## Pré-requisitos

- Python 3.10+ (testado com Python 3.12)
- NumPy
- Scikit-Learn

## Como executar

O script usa somente NumPy e Scikit-Learn. Recomenda-se um ambiente virtual:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scikit-learn
.venv/bin/python rede_neural_iris.py
```

> Observação: se o `pip install` direto no Python do sistema for bloqueado
> (erro "externally-managed-environment"), use o ambiente virtual acima.

## Resultado esperado

```
Shapes dos conjuntos:
X_treino: (120, 4)
X_teste:  (30, 4)
y_treino: (120, 3)
y_teste:  (30, 3)

Acurácia no treino: 97.50%
Acurácia no teste:  96.67%
```

A perda cai de ~1.10 para ~0.05 ao longo das 1000 épocas (`lr = 0.1`, `seed = 42`).

## Atividade

- Nome: Lucas Evaristo Almeida de Paula
- Disciplina: Inteligência Artificial
- Curso: Ciência da Computação
- Instituição: UNIPAC Barbacena
- Professor: Felipe Roncalli de Paula Carneiro
