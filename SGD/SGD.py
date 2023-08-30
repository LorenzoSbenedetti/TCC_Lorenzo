import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time

def prever(nova_matriz, matriz_similaridade, usuario, item, n_vizinhos=10):
    indices_similares = np.argsort(matriz_similaridade[item])[-n_vizinhos-1:-1]
    avaliacoes_similares = nova_matriz[usuario, indices_similares]
    print("avaliacoes similares")
    print(avaliacoes_similares)
    similaridades = matriz_similaridade[item, indices_similares]
    semzero = np.where(avaliacoes_similares > 0)
    avaliacoes_similares = avaliacoes_similares[semzero]
    similaridades = similaridades[semzero]
    numerador = np.sum(avaliacoes_similares * similaridades)
    denominador = np.sum(similaridades)

    if denominador == 0:
        return 0

    return numerador / denominador

if __name__ == '__main__':
    np.random.seed(0)
    d = 50
    df = pd.read_csv('u.csv')

    num_usuarios = df['userId'].nunique() + 1
    num_filmes = df['movieId'].nunique() + 1

    usuario_id_to_idx = {id_: idx + 1 for idx, id_ in enumerate(df['userId'].unique())}
    filme_id_to_idx = {id_: idx + 1 for idx, id_ in enumerate(df['movieId'].unique())}

    P = np.random.normal(size=(num_usuarios, d))
    Q = np.random.normal(size=(num_filmes, d))

    print('DataFrame df (antes do drop):')
    print(df)

    Teste = df.groupby('userId').apply(lambda x: x.sample()).reset_index(drop=True)
    df = df.merge(Teste, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])

    Validacao = df.groupby('userId').apply(lambda x: x.sample()).reset_index(drop=True)
    df = df.merge(Validacao, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])

    min_userId = df['userId'].min()
    min_movieId = df['movieId'].min()

    print('DataFrame Teste:')
    print(Teste)
    print('DataFrame Validação:')
    print(Validacao)

    start_time = time.time()

    num_epochs = 9
    limite_erro = 0.9
    lr = 0.005
    reg = 0.2

    Treino = df.copy()
    Treino = Treino.sample(frac=1).reset_index(drop=True)

    print('Este é o Treino:')
    print(Treino)

    for epoch in range(num_epochs):
        print('Época:', epoch + 1)

        for index, row in Treino.iterrows():
            usuario = usuario_id_to_idx[row['userId']]
            filme = filme_id_to_idx[row['movieId']]
            rating = row['rating']

            erro = rating - np.dot(P[usuario], Q[filme])

            P[usuario] += lr * (erro * Q[filme] - reg * P[usuario])
            Q[filme] += lr * (erro * P[usuario] - reg * Q[filme])

        rmse_validacao = mean_squared_error(
            Validacao['rating'],
            [np.dot(P[usuario_id_to_idx[usuario]], Q[filme_id_to_idx[filme]]) for usuario, filme in
             zip(Validacao['userId'], Validacao['movieId'])],
            squared=False
        )

        print(f'RMSE na validação: {rmse_validacao}')

        if rmse_validacao is not None and rmse_validacao < limite_erro:
            break



    print(f'Fatoração concluída após {epoch + 1} épocas')

    matriz_similaridade = np.dot(Q, Q.T)
    nova_matriz = np.dot(P, Q.T)
    previsoes = [prever(nova_matriz, matriz_similaridade, usuario_id_to_idx[usuario], filme_id_to_idx[filme])
                 for usuario, filme in zip(Teste['userId'], Teste['movieId'])]
    rmse_teste = mean_squared_error(Teste['rating'], previsoes, squared=False)
    print(f'RMSE do Teste: {rmse_teste}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Tempo de execução: {elapsed_time:.2f} segundos')
