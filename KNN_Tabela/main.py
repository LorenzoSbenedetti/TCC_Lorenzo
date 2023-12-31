import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Lendo o arquivo CSV
df = pd.read_csv('ratings.csv')

# Criando uma tabela de classificação de usuários e filmes
ratings = df.pivot_table(index='userId', columns='movieId', values='rating')

# Exibindo a tabela de classificações original
print("Tabela Original de notas:")
print(ratings)

# Criando uma matriz de classificação de usuários e filmes
ratings_matrix = ratings.to_numpy()

# Criando o modelo KNNImputer
imputer = KNNImputer(n_neighbors=5, metric='nan_euclidean')
imputed_ratings_matrix = imputer.fit_transform(ratings_matrix)

# Criando uma tabela de classificação de usuários e filmes com valores imputados
imputed_ratings = pd.DataFrame(imputed_ratings_matrix, index=ratings.index, columns=ratings.columns)

# Exibindo a tabela de classificações imputadas
print("\nMatrix com valores recomendados:")
print(imputed_ratings)

# Salvando a tabela imputada em um arquivo CSV
imputed_ratings.to_csv('imputed_ratings.csv', index=False)
