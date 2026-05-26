
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from metetl.logging_config import get_logger

logger = get_logger()

def read_chunks(filename: str):
    chunks = pd.read_csv(filename, chunksize=10000, usecols=['AccessionYear', 'Object Begin Date'])
    for chunk in chunks: yield chunk

def filter_chunks(chunks):
    for chunk in chunks:
        chunk['AccessionYear'] = pd.to_numeric(chunk['AccessionYear'], errors='coerce')
        chunk['Object Begin Date'] = pd.to_numeric(chunk['Object Begin Date'], errors='coerce')
        chunk['Age'] = chunk['AccessionYear'] - chunk['Object Begin Date']
        chunk['Decade'] = chunk['AccessionYear'] // 10 * 10
        yield chunk[chunk['Age'] >= 0].copy()

def chunk_aggregator(incoming_chunks):
    for chunk in incoming_chunks:
        chunk['Age_sq'] = chunk['Age'] ** 2
        chunk['Count'] = 1
        yield chunk.groupby('Decade')[['Age', 'Age_sq', 'Count']].sum()

def run_analysis_pipeline(csv_path: str):
    logger.info(f"Чтение и анализ данных из {csv_path}...")
    table = pd.DataFrame()
    pipeline = chunk_aggregator(filter_chunks(read_chunks(csv_path)))
    
    for stats in pipeline:
        table = table.add(stats, fill_value=0)
    
    table['Mean_Age'] = table['Age'] / table['Count']
    variance = (table['Age_sq'] / table['Count']) - (table['Mean_Age'] ** 2)
    table['SD'] = np.sqrt(variance.clip(lower=0))
    table['Scatter_Margin'] = 1.96 * table['SD']
    table['Conf_Margin'] = 1.96 * (table['SD'] / np.sqrt(table['Count']))
    table = table.sort_index()
    table['Mean_Diff'] = table['Mean_Age'].diff()
    
    logger.info("Анализ данных завершен.")
    return table

def draw_results(df, output_dir: str):
    logger.info(f"Сохранение графиков в {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    df_plot = df[(df.index >= 1870) & (df.index <= 2020)].copy()
    decades_str = df_plot.index.astype(int).astype(str)

    plt.figure(figsize=(14, 7))
    plt.bar(decades_str, df_plot['Mean_Age'], color='skyblue', edgecolor='black', label='Средний возраст')
    plt.errorbar(decades_str, df_plot['Mean_Age'], yerr=df_plot['Scatter_Margin'], fmt='none', ecolor='red')
    plt.errorbar(decades_str, df_plot['Mean_Age'], yerr=df_plot['Conf_Margin'], fmt='none', ecolor='black')
    plt.title('Анализ возраста объектов при поступлении')
    plt.savefig(os.path.join(output_dir, "age_analysis.png"))
    plt.close()

    df_dyn = df_plot.dropna(subset=['Mean_Diff'])
    dyn_decades = df_dyn.index.astype(int).astype(str)
    
    plt.figure(figsize=(14, 7))
    plt.plot(dyn_decades, df_dyn['Mean_Diff'], marker='o', color='blue')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Динамика изменения среднего возраста объектов')
    plt.savefig(os.path.join(output_dir, "age_dynamics.png"))
    plt.close()
    
    logger.info("Графики успешно сохранены.")