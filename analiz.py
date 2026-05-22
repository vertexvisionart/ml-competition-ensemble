import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import gc

# Настройки
DATA_PATH = "./data"
SAMPLE_SIZE = 50000  # Сколько обычных операций берем для сравнения


def get_data_for_plots():
    print("⏳ Загрузка данных для визуализации...")

    # Загружаем файлы
    train_files = sorted(glob.glob(f"{DATA_PATH}/train_part_*.parquet"))[:1]
    labels_path = f"{DATA_PATH}/train_labels.parquet"

    df = pl.scan_parquet(train_files)
    labels = pl.scan_parquet(labels_path)

    # Джойним метки
    df = df.join(labels, on="event_id", how="left").filter(
        pl.col("target").is_not_null()
    )

    # Базовая подготовка типов
    df = df.with_columns(
        [
            pl.col("event_dttm").str.to_datetime(),
            pl.col("operaton_amt").cast(pl.Float32),
        ]
    ).sort(["customer_id", "event_dttm"])

    # --- ДОБАВЛЕНИЕ НОВЫХ ПАРАМЕТРОВ ТУТ ---
    df = df.with_columns(
        [
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),  # 1-7
            pl.col("event_dttm")
            .diff()
            .dt.total_seconds()
            .over("customer_id")
            .fill_null(999999)
            .alias("seconds_since_prev"),
            # Агрегаты за разные периоды
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="1h", closed="left")
            .over("customer_id")
            .alias("amt_1h"),
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .alias("amt_24h"),
            # Можно добавить количество операций
            pl.len()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .alias("cnt_24h"),
        ]
    )

    # Сбор данных
    fraud = df.filter(pl.col("target") == 1).collect()
    normal_all = df.filter(pl.col("target") == 0).collect()

    actual_sample_n = min(SAMPLE_SIZE, normal_all.height)
    print(
        f"📊 Фрод: {fraud.height}, Обычные: {normal_all.height}. Выборка: {actual_sample_n}"
    )

    normal = normal_all.sample(n=actual_sample_n)

    return fraud.to_pandas(), normal.to_pandas()


def plot_distributions():
    fraud_df, normal_df = get_data_for_plots()

    # --- СПИСОК ПАРАМЕТРОВ ДЛЯ ОТРИСОВКИ ---
    # Формат: (имя_колонки, заголовок_на_русском, использовать_лог_шкалу)
    features_to_plot = [
        ("operaton_amt", "Сумма текущей операции", True),
        ("hour", "Час суток", False),
        ("day_of_week", "День недели (1=Пн, 7=Вс)", False),
        ("seconds_since_prev", "Секунд с прошлой транзакции", True),
        ("amt_1h", "Сумма покупок за последний час", True),
        ("cnt_24h", "Кол-во транзакций за 24 часа", False),
    ]

    # Рассчитываем размер сетки
    n_features = len(features_to_plot)
    n_cols = 2
    n_rows = (n_features + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for i, (col, title, use_log) in enumerate(features_to_plot):
        ax = axes[i]

        if normal_df[col].dropna().empty or fraud_df[col].dropna().empty:
            ax.set_title(f"Нет данных для {title}")
            continue

        # Плотность распределения
        sns.kdeplot(
            data=normal_df,
            x=col,
            ax=ax,
            label="Обычные",
            color="blue",
            fill=True,
            alpha=0.3,
        )
        sns.kdeplot(
            data=fraud_df, x=col, ax=ax, label="Фрод", color="red", fill=True, alpha=0.3
        )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(None)
        ax.set_ylabel("Плотность")
        ax.legend()

        if use_log:
            # Проверка на положительные значения для логарифма
            if (normal_df[col] > 0).any() or (fraud_df[col] > 0).any():
                ax.set_xscale("log")
                ax.set_title(f"{title} (Log Scale)", fontsize=14)

    # Удаляем пустые сабплоты, используя n_features вместо переменной цикла i
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("feature_distributions.png")
    print("✅ Графики успешно обновлены в 'feature_distributions.png'")
    plt.show()


if __name__ == "__main__":
    plot_distributions()

