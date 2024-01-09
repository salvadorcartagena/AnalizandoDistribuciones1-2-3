import pandas as pd
import matplotlib.pyplot as plt

datapd = pd.read_csv('heart_failure_dataset.csv')

def limpieza_categorizacion_datos(data):
    # Verificar valores faltantes
    faltantes = data.isnull().sum().sum()
    if faltantes > 0:
        print(f"¡Hay {faltantes} Datos faltantes en el DataFrame!")
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

anemia_count = data_limpia['anaemia'].value_counts()
diabetes_count = data_limpia['diabetes'].value_counts()
smoking_count = data_limpia['smoking'].value_counts()
death_count = data_limpia['DEATH_EVENT'].value_counts()

colors = ['lightcoral', 'lightseagreen']

plt.figure(figsize=(10, 3))

plt.subplot(1, 4, 1)
plt.pie(anemia_count, labels=['No', 'Si'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Anémicos')

plt.subplot(1, 4, 2)
plt.pie(diabetes_count, labels=['No', 'Si'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Diabéticos')

plt.subplot(1, 4, 3)
plt.pie(smoking_count, labels=['No', 'Si'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Fumadores')

plt.subplot(1, 4, 4)
plt.pie(death_count, labels=['No', 'Si'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Muertos')

plt.tight_layout()
plt.show()