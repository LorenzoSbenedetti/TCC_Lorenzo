import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors

# Lendo o arquivo CSV
df = pd.read_csv('ratings.csv')

# Criando uma tabela de classificação de usuários e filmes
ratings = df.pivot_table(index='userId', columns='movieId', values='rating')

# Substituindo valores NaN por 0
ratings = ratings.fillna(0)

# Displaying the ratings table
print(ratings)

# Criando uma matriz de classificação de usuários e filmes
ratings_matrix = ratings.to_numpy()

# Criando o modelo KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(ratings_matrix)

# Fazendo recomendações para todos os usuários
for user_index in range(ratings_matrix.shape[0]):
    distances, indices = model_knn.kneighbors(ratings_matrix[user_index].reshape(1, -1), n_neighbors=6)
    print('Recomendações para o usuário {0}:\n'.format(ratings.index[user_index]))
    for i in range(1, len(distances.flatten())):
        print('{0}: {1}, com distância de {2}:'.format(i, ratings.index[indices.flatten()[i]], distances.flatten()[i]))
    print('\n')