
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(file_path):

    df = pd.read_csv(file_path)

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values('date')

    print(df.isnull().sum())

    df = df.interpolate(method='linear')

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['Appliances'])
    plt.title("Outlier Detection")
    plt.show()

    return df
