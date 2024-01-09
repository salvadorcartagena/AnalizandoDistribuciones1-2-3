import pandas as pd
import matplotlib.pyplot as plt

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


plt.hist(data_limpia['age'], bins=15, color='peru', edgecolor='black')

plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.title('Distribución de Edades')

plt.show()

male_data = data_limpia[data_limpia['sex'] == 1]  # 1 representa a hombres
female_data = data_limpia[data_limpia['sex'] == 0]  # 0 representa a mujeres

# Calcular la cantidad de anémicos, diabéticos, fumadores y muertos por género
male_counts = [
    male_data['anaemia'].sum(),
    male_data['diabetes'].sum(),
    male_data['smoking'].sum(),
    male_data['DEATH_EVENT'].sum()
]

female_counts = [
    female_data['anaemia'].sum(),
    female_data['diabetes'].sum(),
    female_data['smoking'].sum(),
    female_data['DEATH_EVENT'].sum()
]

categories = ['Anémicos', 'Diabéticos', 'Fumadores', 'Muertos']


bar_width = 0.35
index = range(len(categories))

plt.bar(index, male_counts, bar_width, color='blue', label='Hombres')
plt.bar([i + bar_width for i in index], female_counts, bar_width, color='red', label='Mujeres')


plt.xlabel('Categorías')
plt.ylabel('Cantidad')
plt.title('Histograma Agrupado por sexo')
plt.xticks([i + bar_width / 2 for i in index], categories)
plt.legend()

plt.tight_layout()
plt.show()