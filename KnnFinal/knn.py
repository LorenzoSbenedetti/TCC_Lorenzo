import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import time


def prever(nova_matriz, matriz_similaridade, usuario, item, n_vizinhos=10):
    indices_similares = np.argsort(matriz_similaridade[int(item)].toarray())[0][-n_vizinhos-1:-1]
    avaliacoes_similares = nova_matriz[int(usuario), indices_similares.astype(int)]
    similaridades = matriz_similaridade[int(item), indices_similares.astype(int)]
    avaliacoes_similares = avaliacoes_similares.toarray()
    similaridades = similaridades.toarray()
    numerador = np.sum(avaliacoes_similares * similaridades)
    denominador = np.sum(similaridades)

    if denominador == 0:
        return 0

    return numerador / denominador


if __name__ == '__main__':
    
  
    np.random.seed(0)   
    df = pd.read_csv('ratings.csv')
    df.dropna()
    print('LeituraArquivo')


    df['userId'] = df['userId'].fillna(0).astype(int)
    df['movieId'] = df['movieId'].fillna(0).astype(int)


    min_userId = df['userId'].min()
    min_movieId = df['movieId'].min()


    Teste = df.groupby('userId').apply(lambda x: x.sample()).reset_index(drop=True)
    df = df.merge(Teste, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])  
    print('CriacaoTeste')


    matriz_avaliacoes = csr_matrix((df['rating'], (df['userId'] - min_userId, df['movieId'] - min_movieId)))
    print('MatrizEsparsa')

    num_usuarios = df['userId'].nunique()
    num_filmes = df['movieId'].nunique()

    nova_matriz = matriz_avaliacoes
    print('NovaMatriz')

    print('InicioPrever')


    matriz_similaridade = cosine_similarity(matriz_avaliacoes.T, dense_output=False)
    print('InicioPrevisoes')
    
    start_time = time.time()
    previsoes=[]
    for index, row in Teste.iterrows():
        usuario = row['userId'] - min_userId
        previsoes.append(prever(nova_matriz, matriz_similaridade, usuario, row['movieId'] - min_movieId))

    Teste['userId'] = Teste['userId'].astype(int)
    Teste['movieId'] = Teste['movieId'].astype(int)
    print('finalPrevisao')


    print('inicioRMSE')
    rmse=mean_squared_error(Teste['rating'],previsoes,squared=False)
    print(f'RMSE: {rmse}')



    elapsed_time = time.time() - start_time
    print(f'Tempo de execução: {elapsed_time:.2f} segundos')