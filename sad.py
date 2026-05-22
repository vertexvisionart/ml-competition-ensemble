import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import gc
import numpy as np
import pandas as pd

# Настройки
DATA_PATH = "./data"
SAMPLE_SIZE = 50000  # Сколько обычных операций берем для сравнения

def get_data_for_plots():
    print("⏳ Загрузка данных для визуализации...")
    
    # Загружаем файлы
    train_files = sorted(glob.glob(f"{DATA_PATH}/train_part_*.parquet"))[:1]
    labels_path = f"{DATA_PATH}/train_labels.parquet"
    
    df_lazy = pl.scan_parquet(train_files)
    labels_lazy = pl.scan_parquet(labels_path)
    
    # Джойним метки
    df_lazy = df_lazy.join(labels_lazy, on="event_id", how="left").filter(pl.col("target").is_not_null())
    
    # Сразу определяем все числовые колонки, которые есть в датасете
    # Мы исключаем event_id, customer_id и session_id, так как это просто идентификаторы
    all_cols = df_lazy.collect_schema().names()
    numeric_cols = [
        name for name, dtype in df_lazy.collect_schema().items()
        if dtype.is_numeric() and name not in ['event_id', 'customer_id', 'session_id']
    ]
    
    print(f"📊 Найдено числовых колонок в датасете: {numeric_cols}")

    # Базовая подготовка типов для корректной работы
    df_lazy = df_lazy.with_columns([
        pl.col("event_dttm").str.to_datetime(),
    ]).sort(["customer_id", "event_dttm"])
    
    # Добавляем только самые важные временные параметры (час, день), так как их нет в "сырых" данных
    df_lazy = df_lazy.with_columns([
        pl.col("event_dttm").dt.hour().alias("hour"),
        pl.col("event_dttm").dt.weekday().alias("day_of_week"),
    ])
    
    # Собираем список колонок для итогового DF (оригинальные + таргет + время)
    cols_to_keep = list(set(numeric_cols + ['target', 'hour', 'day_of_week']))
    
    # Сбор данных
    fraud = df_lazy.filter(pl.col("target") == 1).select(cols_to_keep).collect()
    normal_all = df_lazy.filter(pl.col("target") == 0).select(cols_to_keep).collect()
    
    actual_sample_n = min(SAMPLE_SIZE, normal_all.height)
    print(f"📊 Фрод: {fraud.height}, Обычные: {normal_all.height}. Выборка для анализа: {actual_sample_n}")
    
    normal = normal_all.sample(n=actual_sample_n)
    
    return fraud.to_pandas(), normal.to_pandas(), cols_to_keep

def plot_distributions(fraud_df, normal_df, numeric_cols):
    print("📊 Отрисовка распределений основных числовых фич...")
    
    # Берем первые 6 числовых фич для графиков, чтобы не перегружать экран
    features_to_plot = [c for c in numeric_cols if c != 'target'][:6]
    
    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(features_to_plot):
        ax = axes[i]
        
        sns.kdeplot(data=normal_df, x=col, ax=ax, label='Обычные', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(data=fraud_df, x=col, ax=ax, label='Фрод', color='red', fill=True, alpha=0.3)
        
        ax.set_title(f"Распределение: {col}", fontsize=14)
        ax.set_xlabel(None)
        ax.set_ylabel('Плотность')
        ax.legend()
        
        # Автоматический Log Scale для сумм (по названию)
        if 'amt' in col.lower() or 'amount' in col.lower():
            if (normal_df[col] > 0).any():
                ax.set_xscale('log')

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    print("✅ Распределения сохранены в 'feature_distributions.png'")

def plot_correlation_matrix(fraud_df, normal_df, cols_to_corr):
    print("📈 Расчет корреляционной матрицы по датасету...")
    
    # Объединяем фрод и норму
    combined_df = pd.concat([fraud_df, normal_df])
    
    # Оставляем только числовые данные для корреляции
    corr_matrix = combined_df[cols_to_corr].apply(pd.to_numeric, errors='coerce').corr()

    plt.figure(figsize=(14, 12))
    # Рисуем тепловую карту
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
    plt.title('Матрица корреляции оригинальных признаков датасета', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("✅ Матрица корреляции сохранена в 'correlation_matrix.png'")

if __name__ == "__main__":
    fraud_df, normal_df, cols_to_corr = get_data_for_plots()
    
    # Отрисовка распределений
    plot_distributions(fraud_df, normal_df, cols_to_corr)
    
    # Отрисовка матрицы корреляций
    plot_correlation_matrix(fraud_df, normal_df, cols_to_corr)
    
    print("\n🚀 Анализ завершен. Проверь файлы .png в папке со скриптом.")
    plt.show()
