import polars as pl
import numpy as np
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
import glob
import os

# ==========================================
# 1. КОНФИГ
# ==========================================
DATA_PATH = "./data"

CAT_FEATURES = [
    "event_type_nm",
    "channel_indicator_type",
    "channel_indicator_subtype",
    "currency_iso_cd",
    "mcc_code",
    "pos_cd",
    "accept_language",
    "browser_language",
    "timezone",
    "operating_system_type",
    "device_system_version",
    "phone_voip_call_state",
    "web_rdp_connection",
    "compromised",
    "battery",
    "screen_size",
    "developer_tools",
]

# Только необходимые колонки для математики (Train only)
ESSENTIAL_COLS = [
    "event_id",
    "customer_id",
    "event_dttm",
    "operaton_amt",
    "mcc_code",
    "target",
]


# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (БЕЗ ИСТОРИИ, ТОЛЬКО СВЕЖЕЕ)
# ==========================================
def process_data_stable(data_path, is_train=True):
    print(f"🔄 Запуск стабильного пайплайна ({'TRAIN' if is_train else 'TEST'})...")

    if is_train:
        # Train
        files = sorted(glob.glob(f"{data_path}/train_part_*.parquet"))
        df_lazy = pl.scan_parquet(files)

        # Метки
        labels = pl.scan_parquet(f"{data_path}/train_labels.parquet").select(
            ["event_id", "target"]
        )
        df_lazy = df_lazy.join(labels, on="event_id", how="left").with_columns(
            pl.col("target").fill_null(0)
        )

        # Фильтр: только свежие данные (Октябрь 2024+)
        df_lazy = df_lazy.filter(pl.col("event_dttm") >= pl.lit("2024-10-01"))

        # Оптимизация колонок
        cols = [c for c in ESSENTIAL_COLS if c in df_lazy.columns]
        df_lazy = df_lazy.select(cols)

    else:
        # Test
        files_pre = glob.glob(f"{data_path}/*pre*test*.parquet")
        files_test = [
            f
            for f in glob.glob(f"{data_path}/*test*.parquet")
            if "pre" not in os.path.basename(f)
        ]
        if not files_pre:
            files_pre = glob.glob(f"./*pre*test*.parquet")
        if not files_test:
            files_test = [
                f
                for f in glob.glob(f"./*test*.parquet")
                if "pre" not in os.path.basename(f)
            ]

        # Для теста берем Pre-test для контекста
        df_pre = (
            pl.scan_parquet(files_pre[0])
            .select([c for c in ESSENTIAL_COLS if c != "target"])
            .with_columns([pl.lit(0).alias("target"), pl.lit(0).alias("is_submit")])
        )
        df_tst = (
            pl.scan_parquet(files_test[0])
            .select([c for c in ESSENTIAL_COLS if c != "target"])
            .with_columns([pl.lit(0).alias("target"), pl.lit(1).alias("is_submit")])
        )

        df_lazy = pl.concat([df_pre, df_tst], how="vertical")

    # -------------------------------------------------------
    # МАТЕМАТИКА (FE)
    # -------------------------------------------------------
    print("🛠 Генерация признаков...")

    df_lazy = df_lazy.with_columns(pl.col("event_dttm").str.to_datetime())
    df_lazy = df_lazy.sort(["customer_id", "event_dttm"])

    df_lazy = df_lazy.with_columns(
        [
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            (pl.col("event_dttm").dt.hour() < 6).cast(pl.Int8).alias("is_night"),
            # Diff
            pl.col("event_dttm")
            .diff()
            .dt.total_seconds()
            .over("customer_id")
            .fill_null(999999)
            .cast(pl.Float32)
            .alias("seconds_since_prev"),
            pl.col("event_dttm")
            .diff()
            .dt.total_seconds()
            .over(["customer_id", "mcc_code"])
            .fill_null(999999)
            .cast(pl.Float32)
            .alias("seconds_since_prev_mcc"),
            # Rolling 1h & 24h & 7d (Свежие паттерны)
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="1h", closed="left")
            .over("customer_id")
            .cast(pl.Float32)
            .alias("amt_1h"),
            pl.len()
            .rolling("event_dttm", period="1h", closed="left")
            .over("customer_id")
            .cast(pl.UInt16)
            .alias("cnt_1h"),
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .cast(pl.Float32)
            .alias("amt_24h"),
            pl.len()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .cast(pl.UInt16)
            .alias("cnt_24h"),
            pl.col("operaton_amt")
            .mean()
            .rolling("event_dttm", period="7d", closed="left")
            .over("customer_id")
            .cast(pl.Float32)
            .alias("avg_amt_7d"),  # Недельная норма
            # MCC Frequency
            pl.len()
            .rolling("event_dttm", period="24h", closed="left")
            .over(["customer_id", "mcc_code"])
            .cast(pl.UInt16)
            .alias("mcc_count_24h"),
        ]
    )

    # Ratios
    df_lazy = df_lazy.with_columns(
        [
            (pl.col("cnt_1h") / (pl.col("cnt_24h") + 1))
            .cast(pl.Float32)
            .alias("ratio_cnt_1h_24h"),
            (pl.col("operaton_amt") / (pl.col("amt_24h") + 1))
            .cast(pl.Float32)
            .alias("ratio_amt_24h"),
            (pl.col("operaton_amt") / (pl.col("avg_amt_7d") + 1))
            .cast(pl.Float32)
            .alias("ratio_amt_7d"),
        ]
    )

    # Фильтрация
    if is_train:
        print("✂️ Downsampling (10% нормы + весь фрод)...")
        df_lazy = df_lazy.filter(
            (pl.col("target") == 1) | ((pl.col("event_id").hash() % 100) < 10)
        )
    else:
        df_lazy = df_lazy.filter(pl.col("is_submit") == 1)

    print("🚀 Сборка (Collect)...")
    df_math = df_lazy.collect()

    # Подтягиваем категории
    print("🔗 Подтягиваем категории...")
    if is_train:
        files_full = sorted(glob.glob(f"{data_path}/train_part_*.parquet"))
    else:
        files_full = [
            f
            for f in glob.glob(f"{data_path}/*test*.parquet")
            if "pre" not in os.path.basename(f)
        ]
        if not files_full:
            files_full = [
                f
                for f in glob.glob(f"./*test*.parquet")
                if "pre" not in os.path.basename(f)
            ]

    df_cats_lazy = pl.scan_parquet(files_full)
    cols_cats = ["event_id"] + [c for c in CAT_FEATURES if c in df_cats_lazy.columns]

    ids_to_keep = df_math["event_id"]
    df_cats = (
        df_cats_lazy.select(cols_cats)
        .filter(pl.col("event_id").is_in(ids_to_keep))
        .collect()
    )

    df_final = df_math.join(df_cats, on="event_id", how="left")

    for col in CAT_FEATURES:
        if col in df_final.columns:
            df_final = df_final.with_columns(
                pl.col(col).fill_null("MISSING").cast(pl.String)
            )

    return df_final


# ==========================================
# 3. АНСАМБЛЬ (5 МОДЕЛЕЙ)
# ==========================================
if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    gc.collect()

    # 1. Готовим данные (один раз)
    df_train = process_data_stable(DATA_PATH, is_train=True)
    df_test = process_data_stable(DATA_PATH, is_train=False)

    # 2. Готовим Test для Pandas
    print("📦 Конвертация теста в Pandas...")
    features = [
        c
        for c in df_train.columns
        if c
        not in [
            "event_id",
            "customer_id",
            "event_dttm",
            "target",
            "event_desc",
            "is_submit",
        ]
    ]

    X_test = df_test.select(features).to_pandas()
    for col in CAT_FEATURES:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

    # Массив для накопления предсказаний
    final_preds = np.zeros(len(X_test))

    # 3. ЗАПУСК ЦИКЛА (5 SEEDS)
    SEEDS = [42, 777, 2024, 1337, 555]

    print(f"\n🚀 ЗАПУСК АНСАМБЛЯ ({len(SEEDS)} моделей)...")

    for i, seed in enumerate(SEEDS):
        print(f"\nTraining Model {i + 1}/{len(SEEDS)} [Seed {seed}]...")

        # Сортировка Train (каждый раз, т.к. pandas может сбивать индекс)
        df_sorted = df_train.sort("event_dttm")
        pdf = df_sorted.to_pandas()

        for col in CAT_FEATURES:
            if col in pdf.columns:
                pdf[col] = pdf[col].astype(str)

        # Time Split
        split_date = "2025-04-01"
        mask_train = pdf["event_dttm"] < split_date
        mask_val = pdf["event_dttm"] >= split_date

        X_train, y_train = pdf[mask_train][features], pdf[mask_train]["target"]
        X_val, y_val = pdf[mask_val][features], pdf[mask_val]["target"]

        # Модель
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=3,
            task_type="GPU",
            devices="0",
            border_count=32,
            max_ctr_complexity=1,
            cat_features=[c for c in CAT_FEATURES if c in features],
            eval_metric="Logloss",  # Logloss стабильнее для ансамбля
            metric_period=200,
            scale_pos_weight=10,
            early_stopping_rounds=100,
            verbose=200,
            random_seed=seed,  # УНИКАЛЬНЫЙ СИД
            allow_writing_files=False,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        # Предсказание на валидации (для проверки)
        val_score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"🏅 Seed {seed} Local PR-AUC: {val_score:.5f}")

        # Предсказание на тесте и суммирование
        final_preds += model.predict_proba(X_test)[:, 1]

        # Чистим
        del model, pdf, X_train, X_val
        gc.collect()

    # 4. УСРЕДНЕНИЕ
    final_preds /= len(SEEDS)

    # 5. СОХРАНЕНИЕ
    submission = pl.DataFrame({"event_id": df_test["event_id"], "predict": final_preds})

    submission = submission.unique(subset=["event_id"], keep="first")
    submission.write_csv("submission_ensemble.csv")
    print(f"\n✅ ГОТОВО! Файл: submission_ensemble.csv")
