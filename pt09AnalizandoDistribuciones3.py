import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

datapd = pd.read_csv('heart_failure_dataset.csv')

def limpieza_categorizacion_datos(data):
    # Verificar valores faltantes
    faltantes = data.isnull().sum().sum()
    if faltantes > 0:
        print(f"¡Hay {faltantes} datos faltantes en el DataFrame!")
    else:
        print("No hay datos faltantes en el DataFrame.")

    # Verificar filas duplicadas
    duplicados = data.duplicated().sum()
    if duplicados > 0:
        print(f"Hay {duplicados} filas duplicadas en el DataFrame.")
        data.drop_duplicates(inplace=True)
    else:
        print("No hay filas duplicadas en el DataFrame.")

    # Verificar y eliminar valores atípicos
    for column in data.select_dtypes(include='number').columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        filtro_sin_atipicos = (data[column] >= q1 - 1.5 * iqr) & (data[column] <= q3 + 1.5 * iqr)
        data = data[filtro_sin_atipicos]

    return data

data_limpia=limpieza_categorizacion_datos(datapd)


columnas_a_eliminar = ['DEATH_EVENT', 'age']
data_reducida = data_limpia.drop(columns=columnas_a_eliminar)


array_numpy = data_reducida.values


columna_objetivo = data_limpia['DEATH_EVENT'].values
np.savetxt('death_event.csv', columna_objetivo, delimiter=',')

# Reducción de dimensionalidad con t-SNE

X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(array_numpy)

y = columna_objetivo

data = {'Dim_1': X_embedded[:, 0], 'Dim_2': X_embedded[:, 1], 'Dim_3': X_embedded[:, 2], 'DEATH_EVENT': y}
df = pd.DataFrame(data)
color_map = {0: 'Muerto', 1: 'Vivo'}
df['Color'] = df['DEATH_EVENT'].map(color_map)


fig = px.scatter_3d(df, x='Dim_1', y='Dim_2', z='Dim_3', color='Color', 
                    title='Gráfico de Dispersión 3D')


fig.update_traces(marker=dict(size=4, opacity=0.7))
fig.show()