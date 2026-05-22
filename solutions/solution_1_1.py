import polars as pl
import numpy as np
import gc
import glob
import os
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

# ==========================================
# ⚙️ НАСТРОЙКИ
# ==========================================
DATA_PATH = "./data"

# !!! ВАЖНО: ДЛЯ НОУТБУКА СТАВЬ True, ДЛЯ СЕРВЕРА False !!!
DEV_MODE = True

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


def load_and_prepare_full_data(path):
    print("⏳ Загружаем данные (ULTRA LIGHT MODE)...")

    # 1. Сначала ищем все файлы
    files_pre_train = sorted(glob.glob(f"{path}/*pre*train*.parquet"))
    files_train = sorted(glob.glob(f"{path}/train_part_*.parquet"))
    files_pre_test = sorted(glob.glob(f"{path}/*pre*test*.parquet"))
    files_test = [
        f
        for f in glob.glob(f"{path}/*test*.parquet")
        if "pre" not in os.path.basename(f)
    ]

    # === ✂️ ОБРЕЗАЕМ ФАЙЛЫ ДЛЯ НОУТБУКА ===
    if DEV_MODE:
        print("⚠️ DEV_MODE: Берем только ПЕРВЫЙ файл из каждой папки!")
        # Берем только по 1 файлу. Это примерно 33% от данных, а не 100%.
        # Если и это упадет, меняй [:1] на выборку конкретных строк внутри scan_and_tag
        files_pre_train = files_pre_train[:1]
        files_train = files_train[:1]
        # Тест лучше оставить весь, если хочешь сделать сабмишен,
        # но для локальной отладки тоже режем, чтобы не упало.
        files_pre_test = files_pre_test[:1]
        files_test = files_test[:1]

    # Метки
    train_labels = pl.scan_parquet(f"{path}/train_labels.parquet").select(
        [pl.col("event_id").cast(pl.Int64), pl.col("target").cast(pl.Int64)]
    )

    def scan_and_tag(files, dataset_type):
        if not files:
            return None
        try:
            q = pl.scan_parquet(files)
        except:
            q = pl.scan_parquet(files, allow_missing_columns=True)

        # Если DEV_MODE совсем жесткий, можно еще тут отрезать строки
        if DEV_MODE:
            # Берем первые 100,000 строк из файла (Lazy slicing)
            q = q.head(100_000)

        # 1. Фикс типов + session_id
        q = q.with_columns(
            [
                pl.col("event_dttm").str.to_datetime(),
                pl.col("operaton_amt").cast(pl.Float32),
                pl.col("event_id").cast(pl.Int64),
                pl.col("customer_id").cast(pl.Int64),
                # ФИКС session_id (проверка наличия колонки через Schema)
                (pl.col("session_id").cast(pl.Int64).fill_null(-1))
                if "session_id" in q.collect_schema().names()
                else pl.lit(-1, dtype=pl.Int64).alias("session_id"),
            ]
        )

        q = q.with_columns(pl.lit(dataset_type).alias("split_group"))

        if dataset_type == "train":
            q = q.join(train_labels, on="event_id", how="left")
        else:
            q = q.with_columns(pl.lit(None, dtype=pl.Int64).alias("target"))

        return q

    q_pre = scan_and_tag(files_pre_train, "pre_train")
    q_trn = scan_and_tag(files_train, "train")
    q_prt = scan_and_tag(files_pre_test, "pre_test")
    q_tst = scan_and_tag(files_test, "test")

    queries = [x for x in [q_pre, q_trn, q_prt, q_tst] if x is not None]

    # 2. Склеиваем
    full_lazy = pl.concat(queries, how="diagonal")

    # Сортировка - самая тяжелая операция.
    # В Polars она делается в памяти. Если упадет - закомментируй sort,
    # но тогда оконные функции (rolling) будут считать бред.
    full_lazy = full_lazy.sort(["customer_id", "event_dttm"])

    return full_lazy


def generate_features(df_lazy):
    print("🛠 Feature Engineering...")

    schema = df_lazy.collect_schema()
    cols = schema.names()

    # Вспомогательная проверка
    def has(name):
        return name in cols

    res = df_lazy.with_columns(
        [
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            pl.col("event_dttm")
            .diff()
            .dt.total_seconds()
            .over("customer_id")
            .fill_null(999999)
            .alias("seconds_diff"),
            (
                pl.col("event_dttm")
                .diff()
                .dt.total_seconds()
                .over(["customer_id", "mcc_code"])
                .fill_null(999999)
                .alias("seconds_diff_mcc")
                if has("mcc_code")
                else pl.lit(999999).alias("seconds_diff_mcc")
            ),
            # Агрегаты
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="1h", closed="left")
            .over("customer_id")
            .alias("amt_sum_1h"),
            pl.len()
            .rolling("event_dttm", period="1h", closed="left")
            .over("customer_id")
            .alias("cnt_1h"),
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .alias("amt_sum_24h"),
            pl.len()
            .rolling("event_dttm", period="24h", closed="left")
            .over("customer_id")
            .alias("cnt_24h"),
            pl.col("operaton_amt")
            .sum()
            .rolling("event_dttm", period="7d", closed="left")
            .over("customer_id")
            .alias("amt_sum_7d"),
            pl.col("operaton_amt")
            .mean()
            .rolling("event_dttm", period="7d", closed="left")
            .over("customer_id")
            .alias("amt_mean_7d"),
        ]
    )

    # Категории
    for col in CAT_FEATURES:
        if has(col):
            res = res.with_columns(pl.col(col).cast(pl.String).fill_null("MISSING"))
        else:
            res = res.with_columns(pl.lit("MISSING").alias(col))

    return res


def train_model():
    full_lazy = load_and_prepare_full_data(DATA_PATH)
    full_lazy = generate_features(full_lazy)

    print("🚀 Собираем данные в RAM...")
    # На ноуте это теперь займет ~1-2 ГБ вместо 20 ГБ
    df = full_lazy.collect()

    print("✂️ Разделяем...")
    mask_train_valid = df["target"].is_not_null()

    # Train / Val / Test

    # СТАЛО (Исправление):
    # Явно превращаем строку в дату перед сравнением
    split_date = pl.lit("2025-04-01").str.to_datetime()

    mask_train = (
        (df["split_group"] == "train")
        & (df["event_dttm"] < split_date)
        & mask_train_valid
    )
    mask_val = (
        (df["split_group"] == "train")
        & (df["event_dttm"] >= split_date)
        & mask_train_valid
    )
    mask_test = df["split_group"] == "test"

    drop_cols = [
        "event_id",
        "customer_id",
        "event_dttm",
        "target",
        "split_group",
        "event_desc",
    ]
    features = [c for c in df.columns if c not in drop_cols]

    print(f"Признаков: {len(features)}")

    X_train = df.filter(mask_train).select(features).to_pandas()
    y_train = df.filter(mask_train).select("target").to_pandas()

    X_val = df.filter(mask_val).select(features).to_pandas()
    y_val = df.filter(mask_val).select("target").to_pandas()

    X_test = df.filter(mask_test).select(features).to_pandas()
    test_ids = df.filter(mask_test).select("event_id").to_series().to_list()

    del df, full_lazy
    gc.collect()

    print(f"Размер Train: {X_train.shape}, Val: {X_val.shape}")

    print("🔥 Обучение (Fast Mode)...")
    # Для теста ставим меньше итераций
    iters = 500 if DEV_MODE else 3000

    model = CatBoostClassifier(
        iterations=iters,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        # Если на ноуте нет GPU, CatBoost сам переключится на CPU, но лучше явно указать
        # task_type="GPU" if not DEV_MODE else "CPU",
        task_type="CPU",  # Пока давай на CPU для надежности на ноуте
        cat_features=[c for c in CAT_FEATURES if c in X_train.columns],
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    preds_val = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, preds_val)
    print(f"\n✅ VALIDATION PR-AUC: {pr_auc:.5f}")

    print("📝 Сабмишен...")
    preds_test = model.predict_proba(X_test)[:, 1]
    submission = pl.DataFrame({"event_id": test_ids, "predict": preds_test})
    submission = submission.unique(subset=["event_id"], keep="first")
    submission.write_csv("submission_dev.csv")
    print("🎉 Готово! Файл: submission_dev.csv")


if __name__ == "__main__":
    train_model()
