import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_chunks(filename: str):
    chunks = pd.read_csv(filename, chunksize=10000,usecols=['AccessionYear', 'Object Begin Date'])
    for chunk in chunks:
        yield chunk
 

def filter_chunks(chunks):
    for chunk in chunks:
        chunk['AccessionYear'] = pd.to_numeric(chunk['AccessionYear'], errors='coerce')
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'],errors='coerce')

        chunk['Age'] = chunk['AccessionYear'] - chunk['Object Begin Date']
        chunk['Decade'] = chunk['AccessionYear'] // 10 * 10

        chunk = chunk[chunk['Age'] >= 0].copy()

        yield chunk
    


def chunk_aggregator(incoming_chunks):
    for chunk in incoming_chunks:

        chunk['Age_sq'] = chunk['Age'] ** 2
        chunk['Count'] = 1

        yield chunk.groupby('Decade')[['Age', 'Age_sq', 'Count']].sum()



def run_analysis_pipeline(pipeline):

    table = pd.DataFrame()
    
    for stats in pipeline:
        table = table.add(stats, fill_value=0)
    

    table['Mean_Age'] = table['Age'] / table['Count']
    
    variance = (table['Age_sq'] / table['Count']) - (table['Mean_Age'] ** 2)
    table['SD'] = np.sqrt(variance.clip(lower=0))
    
    table['Scatter_Margin'] = 1.96 * table['SD']
    table['Conf_Margin'] = 1.96 * (table['SD'] / np.sqrt(table['Count']))
    
    table = table.sort_index()
    table['Mean_Diff'] = table['Mean_Age'].diff()
    
    return table


def draw_results(df):

    df_plot = df[(df.index >= 1870) & (df.index <= 2020)].copy()
    decades_str = df_plot.index.astype(int).astype(str)

    plt.figure(figsize=(14, 7))
    plt.bar(decades_str, df_plot['Mean_Age'], color='skyblue', edgecolor='black', label='Средний возраст')
    
    plt.errorbar(decades_str, df_plot['Mean_Age'], yerr=df_plot['Scatter_Margin'], fmt='none', 
                 ecolor='red', capsize=5, elinewidth=1, label='95% Интервал рассеяния')
    
    plt.errorbar(decades_str, df_plot['Mean_Age'], yerr=df_plot['Conf_Margin'], fmt='none', 
                 ecolor='black', capsize=3, elinewidth=3, label='95% Доверительный интервал')

    plt.title('Анализ возраста объектов при поступлении')
    plt.xlabel('Десятилетие поступления')
    plt.ylabel('Возраст (лет)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    df_dyn = df_plot.dropna(subset=['Mean_Diff'])
    dyn_decades = df_dyn.index.astype(int).astype(str)

    plt.figure(figsize=(14, 7))
    
    plt.plot(dyn_decades, df_dyn['Mean_Diff'], marker='o', linestyle='-', 
             color='blue', linewidth=2, label='Разница с прошлым десятилетием')
    
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    rolling_trend = df_dyn['Mean_Diff'].rolling(window=3, min_periods=1).mean()
    plt.plot(dyn_decades, rolling_trend, color='red', linestyle=':', label='Тренд (скользящее среднее)')

    plt.title('Динамика изменения среднего возраста объектов')
    plt.xlabel('Десятилетие')
    plt.ylabel('Изменение возраста (лет)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == '__main__':
  
    final = run_analysis_pipeline(chunk_aggregator(filter_chunks(read_chunks('MetObjects.csv'))))  

    draw_results(final)

