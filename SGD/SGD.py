import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time

if __name__ == '__main__':
    np.random.seed(0)
    d = 10
    df = pd.read_csv('ratings.csv')

    num_usuarios = df['userId'].nunique() + 1
    num_filmes = df['movieId'].nunique() + 1

    usuario_id_to_idx = {id_: idx + 1 for idx, id_ in enumerate(df['userId'].unique())}
    filme_id_to_idx = {id_: idx + 1 for idx, id_ in enumerate(df['movieId'].unique())}

    P = np.random.normal(0, 0.1, size=(num_usuarios, d))
    Q = np.random.normal(0, 0.1, size=(num_filmes, d))

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

    num_epochs = 15
    lr = 0.01
    reg = 0.005

    Treino = df.copy()
    Treino = Treino.sample(frac=1).reset_index(drop=True)

    print('Este é o Treino:')
    print(Treino)

    # Abre o arquivo para escrita
    with open('Resultados.txt', 'w') as arquivo_resultados:
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

            # Escreve os resultados da validação no arquivo
            arquivo_resultados.write(f'Época: {epoch + 1}, RMSE na validação: {rmse_validacao}\n')

            if rmse_validacao == 0:
                break

        print(f'Fatoração concluída após {epoch + 1} épocas')

        previsoes = [np.dot(P[usuario_id_to_idx[usuario]], Q[filme_id_to_idx[filme]])
                     for usuario, filme in zip(Teste['userId'], Teste['movieId'])]
        rmse_teste = mean_squared_error(Teste['rating'], previsoes, squared=False)

        # Escreve o resultado do RMSE de teste no arquivo
        arquivo_resultados.write(f'RMSE do Teste: {rmse_teste}\n')

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Escreve o tempo de execução no arquivo
        arquivo_resultados.write(f'Tempo de execução: {elapsed_time:.2f} segundos\n')

    print(f'Tempo de execução: {elapsed_time:.2f} segundos')
