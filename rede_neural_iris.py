# Nome: Lucas Evaristo Almeida de Paula
# Disciplina: Inteligência Artificial
# Curso: Ciência da Computação
# Instituição: UNIPAC Barbacena
# Professor: Felipe Roncalli de Paula Carneiro
# Data: 05/11/2026
# Atividade: Rede Neural Artificial (camada escondida) do zero com NumPy para classificar o dataset Iris

# rede_neural_iris.py
# Rede neural do zero (NumPy + Scikit-Learn apenas para carga/preparação dos dados)
# Script completo: pré-processamento, treinamento (gradiente descendente) e avaliação da rede.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Semente fixa para reprodutibilidade
RANDOM_STATE = 42


def carregar_e_preparar_dados():
    """Carrega o Iris, divide treino/teste, padroniza as features e aplica
    One-Hot Encoding nos rótulos.

    Obs.: a ordem difere um pouco do enunciado, mas por um motivo importante:
    dividir ANTES de padronizar evita vazamento de dados (data leakage), pois
    o scaler é ajustado somente com o conjunto de treino.

    Retorna:
        X_treino (ndarray): features de treino padronizadas, shape (120, 4)
        X_teste  (ndarray): features de teste padronizadas, shape (30, 4)
        y_treino (ndarray): rótulos de treino em One-Hot, shape (120, 3)
        y_teste  (ndarray): rótulos de teste em One-Hot, shape (30, 3)
    """
    # 1. Carregar o dataset Iris (150 amostras, 4 features, 3 classes)
    iris = load_iris()
    X = iris.data      # features originais (não padronizadas), shape (150, 4)
    y = iris.target    # rótulos inteiros (0, 1, 2), shape (150,)

    # 2. Dividir em treino (80%) e teste (20%)
    #    stratify=y (rótulos inteiros) mantém a proporção das classes nos dois
    #    conjuntos. Feito ANTES da padronização para não vazar dados do teste
    #    para o treino. random_state fixo garante reprodutibilidade.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Padronizar as features: média 0 e desvio padrão 1
    #    O scaler é ajustado (fit) APENAS com o treino e a mesma transformação
    #    é aplicada ao teste (transform). Assim, o teste não "informa" sua
    #    média/desvio ao treino, evitando vazamento de dados.
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)  # aprende média/desvio do treino
    X_teste = scaler.transform(X_teste)        # só aplica a transformação ao teste

    # 4. Converter os rótulos para One-Hot Encoding (vetores de tamanho 3)
    #    Ex.: classe 0 -> [1, 0, 0], classe 1 -> [0, 1, 0], classe 2 -> [0, 0, 1]
    #    Convertido após a divisão, separadamente para treino e teste
    y_treino = np.eye(3)[y_treino]
    y_teste = np.eye(3)[y_teste]

    return X_treino, X_teste, y_treino, y_teste


# ---------------------------------------------------------------------------
# FUNÇÕES DE ATIVAÇÃO E FUNÇÃO DE PERDA (somente NumPy)
# ---------------------------------------------------------------------------

def relu(z):
    """Aplica a função de ativação ReLU (Rectified Linear Unit).

    Substitui todos os valores negativos por zero e mantém os positivos.
    Trabalha elemento a elemento, então aceita arrays de qualquer shape
    (vetor, matriz etc.). Serve como não-linearidade nas camadas ocultas.
    """
    return np.maximum(0, z)


def relu_derivada(z):
    """Calcula a derivada da ReLU em relação a z.

    Vale 1 onde z > 0 (região linear) e 0 onde z <= 0 (região saturada),
    elemento a elemento. É usada no backpropagation para propagar o
    gradiente pela camada oculta.
    """
    return (z > 0).astype(z.dtype if np.issubdtype(z.dtype, np.floating) else float)


def softmax(z):
    """Converte logits em probabilidades aplicando a função Softmax por linha.

    Versão numericamente estável: subtrai o máximo de cada linha antes do
    exp(), evitando overflow em logits grandes. O resultado de cada linha
    soma exatamente 1, formando uma distribuição de probabilidade sobre as
    classes. Pensada para lotes: cada linha de z é uma amostra.
    """
    z_deslocado = z - np.max(z, axis=1, keepdims=True)  # deslocamento por linha
    exp_z = np.exp(z_deslocado)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(y_real, y_previsto):
    """Calcula a perda média do lote usando entropia cruzada categórica.

    y_real deve estar em One-Hot Encoding (mesmo shape de y_previsto).
    As probabilidades em y_previsto passam por clipping em [1e-12, 1] antes
    do log(), evitando log(0). A soma é feita sobre as classes e a média
    sobre as amostras do lote. Quanto menor o valor, melhor a previsão.
    """
    y_previsto = np.clip(y_previsto, 1e-12, 1.0)  # evita log(0)
    perdas = -np.sum(y_real * np.log(y_previsto), axis=1)
    return np.mean(perdas)


def inicializar_parametros(n_entrada=4, n_escondida=8, n_saida=3, seed=42):
    """Inicializa os pesos e vieses da rede neural.

    Arquitetura da rede: n_entrada -> n_escondida (ReLU) -> n_saida (Softmax).

    Pesos (valores aleatórios pequenos, para evitar saturação):
      W1: shape (n_entrada, n_escondida) = (4, 8)
          Inicialização He (adequada à ReLU): randn * sqrt(2 / n_entrada).
      W2: shape (n_escondida, n_saida) = (8, 3)
          Valores pequenos: randn * 0.01.
    Vieses (iniciados em zero, como de costume):
      b1: shape (1, n_escondida) = (1, 8)
      b2: shape (1, n_saida)     = (1, 3)

    A semente `seed` torna a inicialização reproduzível.
    Retorna um dicionário com as chaves 'W1', 'b1', 'W2' e 'b2'.
    """
    rng = np.random.default_rng(seed)  # gerador com semente fixa

    parametros = {
        # W1: (4, 8) - pesos da camada de entrada para a escondida (He init)
        'W1': rng.standard_normal((n_entrada, n_escondida)) * np.sqrt(2.0 / n_entrada),
        # b1: (1, 8) - viés da camada escondida (será transmitido via broadcasting)
        'b1': np.zeros((1, n_escondida)),
        # W2: (8, 3) - pesos da camada escondida para a saída
        'W2': rng.standard_normal((n_escondida, n_saida)) * 0.01,
        # b2: (1, 3) - viés da camada de saída
        'b2': np.zeros((1, n_saida)),
    }
    return parametros


def forward(X, parametros):
    """Propagação para frente (forward pass) da rede neural.

    X: shape (n_amostras, n_entrada) = (lote, 4)

    Etapas (shapes considerando lote de m amostras):
      Z1 = X·W1 + b1  -> (m, 4)·(4, 8) + (1, 8) = (m, 8)  # pré-ativação 1
      A1 = relu(Z1)   -> (m, 8)                           # ativação oculta
      Z2 = A1·W2 + b2 -> (m, 8)·(8, 3) + (1, 3) = (m, 3)  # pré-ativação 2 (logits)
      A2 = softmax(Z2)-> (m, 3)                           # probabilidades

    Retorna (A2, cache), onde:
      A2:    (m, 3) - probabilidades previstas por classe
      cache: dicionário com 'Z1' (m, 8), 'A1' (m, 8) e 'A2' (m, 3),
             guardados para serem reutilizados no backpropagation.
    """
    W1, b1 = parametros['W1'], parametros['b1']
    W2, b2 = parametros['W2'], parametros['b2']

    Z1 = X @ W1 + b1     # (m, 4)·(4, 8) + (1, 8) -> (m, 8)
    A1 = relu(Z1)        # (m, 8)
    Z2 = A1 @ W2 + b2    # (m, 8)·(8, 3) + (1, 3) -> (m, 3)
    A2 = softmax(Z2)     # (m, 3) - cada linha soma 1

    cache = {'Z1': Z1, 'A1': A1, 'A2': A2}  # valores salvos para o backward pass
    return A2, cache


def backward(X, y, parametros, cache):
    """Backpropagation: calcula os gradientes da perda em relação aos parâmetros.

    Regra da cadeia combinada softmax + entropia cruzada:
    a derivada da perda em relação aos logits Z2 tem forma fechada (A2 - y),
    o que simplifica bastante o cálculo.

    X: shape (m, 4) - features do lote (m amostras)
    y: shape (m, 3) - rótulos em One-Hot Encoding

    Passos (com shapes):
      m    = número de amostras no lote
      dZ2  = (A2 - y) / m        -> (m, 3)  # erro na saída, média sobre o lote
      dW2  = A1.T · dZ2          -> (8, m)·(m, 3) = (8, 3)  # quanto W2 contribuiu
      db2  = soma(dZ2, eixo 0)   -> (1, 3)  # gradiente do viés de saída
      dA1  = dZ2 · W2.T          -> (m, 3)·(3, 8) = (m, 8)  # erro que volta p/ oculta
      dZ1  = dA1 * relu_derivada(Z1)          -> (m, 8)  # "mascara" onde Z1 <= 0 (ReLU)
      dW1  = X.T · dZ1           -> (4, m)·(m, 8) = (4, 8)  # quanto W1 contribuiu
      db1  = soma(dZ1, eixo 0)   -> (1, 8)  # gradiente do viés oculto

    Retorna um dicionário com dW1, db1, dW2 e db2.
    """
    m = X.shape[0]  # número de amostras no lote

    # Recupera os valores salvos no forward e os parâmetros
    Z1, A1, A2 = cache['Z1'], cache['A1'], cache['A2']
    W2 = parametros['W2']

    # 1) Erro na camada de saída (softmax + cross-entropy já simplificadas):
    #    derivada d(perda)/dZ2 = A2 - y; dividimos por m pois a perda é a MÉDIA do lote
    dZ2 = (A2 - y) / m                                  # (m, 3)

    # 2) Gradientes de W2 e b2: regra da cadeia dL/dW = entrada.T · dL/dZ
    dW2 = A1.T @ dZ2                                    # (8, m)·(m, 3) -> (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True)            # (1, 3) - mesma shape de b2

    # 3) Propaga o erro de volta para a camada escondida:
    #    dL/dA1 = dZ2 · W2.T (transpõe os pesos para "desfazer" a multiplicação do forward)
    dA1 = dZ2 @ W2.T                                    # (m, 3)·(3, 8) -> (m, 8)
    #    Aplica a derivada da ReLU: onde Z1 <= 0 o gradiente é zerado
    dZ1 = dA1 * relu_derivada(Z1)                       # (m, 8)

    # 4) Gradientes de W1 e b1, mesma lógica de (2), agora com a entrada X
    dW1 = X.T @ dZ1                                     # (4, m)·(m, 8) -> (4, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True)            # (1, 8) - mesma shape de b1

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def atualizar_parametros(parametros, gradientes, taxa_aprendizado):
    """Atualiza os parâmetros com gradiente descendente (um passo).

    Regra: parametro = parametro - taxa_aprendizado * gradiente.
    Os gradientes vêm com as chaves dW1, db1, dW2, db2 e são casados com
    W1, b1, W2, b2 automaticamente (prefixo 'd').
    """
    parametros_novo = {}
    for nome, valor in parametros.items():
        grad = gradientes['d' + nome]          # ex.: 'W1' -> gradiente 'dW1'
        parametros_novo[nome] = valor - taxa_aprendizado * grad
    return parametros_novo


def treinar(X, y, epocas=1000, taxa_aprendizado=0.1):
    """Treina a rede completa: forward -> perda -> backward -> atualização.

    Executa o loop de treino por `epocas` iterações em todo o dataset
    (gradiente descendente em lote/batch). Imprime a perda a cada 100
    épocas para acompanhar a convergência.

    Retorna (parametros_treinados, historico_perdas), onde
    historico_perdas é a lista da perda de cada época.
    """
    # Inicializa os pesos; n_entrada e n_saida são deduzidos dos dados
    parametros = inicializar_parametros(
        n_entrada=X.shape[1], n_escondida=8, n_saida=y.shape[1], seed=42
    )

    historico_perdas = []  # guarda a perda de cada época

    for epoca in range(1, epocas + 1):
        # 1) Forward: calcula as probabilidades previstas
        A2, cache = forward(X, parametros)
        # 2) Perda: quanto o modelo errou nesta época
        perda = cross_entropy(y, A2)
        historico_perdas.append(perda)
        # 3) Backward: gradientes da perda em relação a cada parâmetro
        gradientes = backward(X, y, parametros, cache)
        # 4) Atualização: um passo de gradiente descendente
        parametros = atualizar_parametros(parametros, gradientes, taxa_aprendizado)

        if epoca % 100 == 0 or epoca == epocas:
            print(f"Época {epoca}/{epocas} - perda (loss): {perda:.6f}")

    return parametros, historico_perdas


def prever(X, parametros):
    """Faz a inferência: retorna a CLASSE prevista para cada amostra.

    Executa o forward pass (A2 = probabilidades de cada classe) e devolve
    o índice da classe de maior probabilidade (argmax no eixo 1).
    X: shape (m, 4); retorna vetor shape (m,) com valores em {0, 1, 2}.
    """
    A2, _ = forward(X, parametros)  # A2: (m, 3) - probabilidades por classe
    return np.argmax(A2, axis=1)    # (m,) - classe com maior probabilidade


def calcular_acuracia(y_real_onehot, y_previsto):
    """Calcula a acurácia do modelo.

    Compara as classes reais (argmax dos rótulos One-Hot) com as classes
    previstas. Aceita y_previsto já como vetor de classes (shape (m,)) ou
    como probabilidades (shape (m, 3)), aplicando argmax neste último caso.
    Retorna um float em [0, 1].
    """
    classes_reais = np.argmax(y_real_onehot, axis=1)      # (m,)
    y_previsto = np.asarray(y_previsto)
    if y_previsto.ndim > 1:                               # veio como probabilidades
        y_previsto = np.argmax(y_previsto, axis=1)        # (m,)
    return np.mean(classes_reais == y_previsto)


if __name__ == "__main__":
    # 1) Carrega e prepara os dados (Iris padronizado, one-hot, split 80/20)
    X_treino, X_teste, y_treino, y_teste = carregar_e_preparar_dados()

    # Imprime os shapes dos conjuntos para conferência
    print("Shapes dos conjuntos:")
    print(f"X_treino: {X_treino.shape}")  # (120, 4)
    print(f"X_teste:  {X_teste.shape}")   # (30, 4)
    print(f"y_treino: {y_treino.shape}")  # (120, 3)
    print(f"y_teste:  {y_teste.shape}")   # (30, 3)

    # 2) Inicializa os parâmetros da rede (exibição didática da inicialização;
    #    o treinar() reutiliza a mesma inicialização, pois ambos usam seed=42)
    parametros_iniciais = inicializar_parametros(n_entrada=4, n_escondida=8, n_saida=3, seed=42)
    print("\nParâmetros inicializados:")
    for nome, matriz in parametros_iniciais.items():
        print(f"  {nome}: shape {matriz.shape}")

    # 3) Treina a rede por 1000 épocas (forward + loss + backward + atualização)
    print("\nTreinando a rede...")
    parametros, historico_perdas = treinar(X_treino, y_treino, epocas=1000, taxa_aprendizado=0.1)

    # 4) Inferência no conjunto de teste (dados que o modelo nunca viu)
    y_previsto = prever(X_teste, parametros)

    # 5) Acurácia final em porcentagem
    acuracia_teste = calcular_acuracia(y_teste, y_previsto)
    acuracia_treino = calcular_acuracia(y_treino, prever(X_treino, parametros))
    print(f"\nAcurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste:  {acuracia_teste * 100:.2f}%")
