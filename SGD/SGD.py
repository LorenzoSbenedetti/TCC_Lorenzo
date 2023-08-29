import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import time

if __name__ == '__main__':
    np.random.seed(0)   
    d = 100
    df = pd.read_csv('u.csv')

    num_usuarios = df['userId'].nunique()
    num_filmes = df['movieId'].nunique()
    
    usuario_id_to_idx = {id_: idx for idx, id_ in enumerate(df['userId'].unique())}
    filme_id_to_idx = {id_: idx for idx, id_ in enumerate(df['movieId'].unique())}
    
    P = np.random.normal(size=(num_usuarios, d))
    Q = np.random.normal(size=(num_filmes, d))
    
    print('DataFrame df (antes do drop):')
    print(df)
    
    Teste = df.groupby('userId').apply(lambda x: x.sample()).reset_index(drop=True)
    df = df.merge(Teste, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])  
   
   
    #print (pd.merge(df,Teste, indicator=True, how='outer')
    #     .query('_merge=="left_only"')
    #     .drop('_merge', axis=1))
   
    Validacao = df.groupby('userId').apply(lambda x: x.sample()).reset_index(drop=True)
    df = df.merge(Validacao, how='outer', indicator=True)
    df = df[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])
    
    #print (pd.merge(df,Validacao, indicator=True, how='outer')
    #     .query('_merge=="left_only"')
    #     .drop('_merge', axis=1))
    
    min_userId = df['userId'].min()
    min_movieId = df['movieId'].min()

    print('DataFrame Teste:')
    print(Teste)
    print('DataFrame Validação:')
    print(Validacao)
    
    start_time = time.time()
    
    num_epochs = 9
    limite_erro = 0.9
    lr =  0.001
    reg = 0.05

    Treino= df.copy()
    print('Este e o Treino :')
    print(Treino)
    
    for epoch in range(num_epochs):
        print('Época:', epoch + 1)
        
        for index, row in df.iterrows():
            usuario = usuario_id_to_idx[row['userId']]
            filme = filme_id_to_idx[row['movieId']]
            rating = row['rating']
            
            erro = rating - np.dot(P[usuario], Q[filme])
            
            P[usuario] += lr * ( erro * Q[filme] - reg * P[usuario])
            Q[filme] += lr * ( erro * P[usuario] - reg * Q[filme])
        
        rmse_validacao = mean_squared_error(
            Validacao['rating'],
            [np.dot(P[usuario_id_to_idx[usuario]], Q[filme_id_to_idx[filme]]) for usuario, filme in zip(Validacao['userId'], Validacao['movieId'])],
            squared=False
        )
        
        print(f'RMSE na validação: {rmse_validacao}')
        
        if rmse_validacao is not None and rmse_validacao < limite_erro:
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f'Fatoração concluída após {epoch+1} épocas')
    print(f'Tempo de execução: {elapsed_time:.2f} segundos')
