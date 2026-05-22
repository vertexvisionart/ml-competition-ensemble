import polars as pl
import numpy as np
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
import glob
import os
import time

# ==========================================
# 1. КОНФИГ
# ==========================================
DATA_PATH = "./data"
DEV_MODE = True  # True для ноутбука (берет 5% юзеров), False для сервера (берет все)

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
    "session_id"
]


# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ
# ==========================================
def process_data_stable(data_path, is_train=True):
    start_time = time.time()
    mode_name = 'TRAIN' if is_train else 'TEST'
    print(f"🔄 Запуск пайплайна ({mode_name})... DEV_MODE={DEV_MODE}")

    # Вспомогательная функция для безопасного чтения
    def scan_safe(files):
        if not files: return None
        try:
            q = pl.scan_parquet(files)
        except:
            q = pl.scan_parquet(files, allow_missing_columns=True)
        
        # --- ФИКС ТИПОВ ---
        schema = q.collect_schema().names()
        
        cols_to_cast = [
            (pl.col("event_dttm").str.to_datetime()),
            (pl.col("operaton_amt").cast(pl.Float32)),
            (pl.col("event_id").cast(pl.Int64)),
            (pl.col("customer_id").cast(pl.Int64))
        ]
        
        if "session_id" in schema:
            cols_to_cast.append(pl.col("session_id").cast(pl.Int64).fill_null(-1))
        else:
            cols_to_cast.append(pl.lit(-1, dtype=pl.Int64).alias("session_id"))
            
        return q.with_columns(cols_to_cast)

    if is_train:
        # Грузим и Train, и Pre-train (чтобы была история)
        files_train = sorted(glob.glob(f"{data_path}/train_part_*.parquet"))
        files_pre = sorted(glob.glob(f"{data_path}/*pre*train*.parquet"))
        
        # В DEV_MODE можно брать не все файлы, но лучше фильтровать по customer_id позже,
        # чтобы точно сохранить целостность истории конкретного человека.
        # Если памяти совсем мало, можно раскомментировать slicing:
        if DEV_MODE:
             files_train = files_train[:2] # Берем чуть больше, чтобы точно попасть в даты
             files_pre = files_pre[:2]
        
        q_train = scan_safe(files_train)
        q_pre = scan_safe(files_pre)
        
        # Склеиваем
        parts = [x for x in [q_pre, q_train] if x is not None]
        if not parts: raise ValueError("❌ Нет файлов для обучения!")
        df_lazy = pl.concat(parts, how="diagonal")

        # Метки
        labels = pl.scan_parquet(f"{data_path}/train_labels.parquet").select(
            ["event_id", "target"]
        )
        
        df_lazy = df_lazy.join(labels, on="event_id", how="left")

        # Оптимизация колонок
        schema = df_lazy.collect_schema().names()
        cols = [c for c in ESSENTIAL_COLS if c in schema]
        df_lazy = df_lazy.select(cols)

    else:
        # === ПОЛНАЯ ИСТОРИЯ ДЛЯ ТЕСТА ===
        print("📥 Загрузка истории для тестовых клиентов...")
        
        files_pre_test = sorted(glob.glob(f"{data_path}/*pre*test*.parquet"))
        files_test = sorted([f for f in glob.glob(f"{data_path}/*test*.parquet") if "pre" not in os.path.basename(f)])
        
        if DEV_MODE:
            files_test = files_test[:1] # Только кусочек теста

        # 1. Загружаем сам тест, чтобы узнать ID клиентов
        q_test = scan_safe(files_test)
        if q_test is None: raise ValueError("Нет тестовых файлов!")
        
        # Получаем список клиентов теста
        test_customers = q_test.select("customer_id").unique().collect().to_series()
        print(f"👥 Найдено {len(test_customers)} клиентов в тесте.")

        # 2. Загружаем ВСЮ историю
        files_history = sorted(glob.glob(f"{data_path}/*train*.parquet")) + files_pre_test
        # В DEV_MODE историю тоже можно подрезать, если файлы огромные
        if DEV_MODE: files_history = files_history[:3]

        q_history = scan_safe(files_history)
        
        # Фильтруем историю только по тестовым клиентам
        if q_history is not None:
            q_history = q_history.filter(pl.col("customer_id").is_in(test_customers))
            q_history = q_history.select([c for c in ESSENTIAL_COLS if c != "target"]).with_columns([
                pl.lit(None, dtype=pl.Int64).alias("target"), 
                pl.lit(0).alias("is_submit")
            ])
        
        # Подготавливаем сам тест
        q_test = q_test.select([c for c in ESSENTIAL_COLS if c != "target"]).with_columns([
            pl.lit(None, dtype=pl.Int64).alias("target"), 
            pl.lit(1).alias("is_submit")
        ])
        
        # Объединяем
        parts = []
        if q_history is not None: parts.append(q_history)
        parts.append(q_test)
        
        df_lazy = pl.concat(parts, how="diagonal")

    # -------------------------------------------------------
    # ФИЛЬТРАЦИЯ DEV_MODE (УМНАЯ)
    # -------------------------------------------------------
    # Вместо head(), который режет всё подряд, берем % пользователей.
    # Это сохраняет историю целиком для выбранных людей.
    if DEV_MODE:
        print(f"⚠️ DEV_MODE: Оставляем только 5% пользователей (по хэшу ID)...")
        # Берем остаток от деления хэша на 20 (это 1/20 часть = 5%)
        # Это гарантирует, что и train, и история одного юзера останутся вместе
        df_lazy = df_lazy.filter((pl.col("customer_id").hash() % 20) == 0)

    # -------------------------------------------------------
    # МАТЕМАТИКА (FE)
    # -------------------------------------------------------
    print("🛠 Генерация признаков...")

    # УДАЛЕНО: df_lazy.with_columns(pl.col("event_dttm").str.to_datetime())
    # Причина: типы уже исправлены в scan_safe
    df_lazy = df_lazy.sort(["customer_id", "event_dttm"])

    schema = df_lazy.collect_schema().names()
    has_mcc = "mcc_code" in schema
    has_session = "session_id" in schema

    # Базовые признаки
    df_lazy = df_lazy.with_columns([
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            (pl.col("event_dttm").dt.hour() < 6).cast(pl.Int8).alias("is_night"),
            
            pl.col("event_dttm").diff().dt.total_seconds().over("customer_id").fill_null(999999).cast(pl.Float32).alias("seconds_since_prev"),
            
            (pl.col("event_dttm").diff().dt.total_seconds().over(["customer_id", "mcc_code"]).fill_null(999999).cast(pl.Float32).alias("seconds_since_prev_mcc")
             if has_mcc else pl.lit(999999).alias("seconds_since_prev_mcc")),
    ])
    
    # Окна
    rolling_ops = [
        # 1h
        pl.col("operaton_amt").sum().rolling("event_dttm", period="1h", closed="left").over("customer_id").cast(pl.Float32).alias("amt_1h"),
        pl.len().rolling("event_dttm", period="1h", closed="left").over("customer_id").cast(pl.UInt16).alias("cnt_1h"),
        # 24h
        pl.col("operaton_amt").sum().rolling("event_dttm", period="24h", closed="left").over("customer_id").cast(pl.Float32).alias("amt_24h"),
        pl.len().rolling("event_dttm", period="24h", closed="left").over("customer_id").cast(pl.UInt16).alias("cnt_24h"),
        # 7d
        pl.col("operaton_amt").mean().rolling("event_dttm", period="7d", closed="left").over("customer_id").cast(pl.Float32).alias("avg_amt_7d"),
        # 30d
        pl.col("operaton_amt").sum().rolling("event_dttm", period="30d", closed="left").over("customer_id").cast(pl.Float32).alias("amt_30d"),
        pl.len().rolling("event_dttm", period="30d", closed="left").over("customer_id").cast(pl.UInt16).alias("cnt_30d"),
    ]
    
    if has_mcc:
        rolling_ops.append(pl.len().rolling("event_dttm", period="24h", closed="left").over(["customer_id", "mcc_code"]).cast(pl.UInt16).alias("mcc_count_24h"))
        rolling_ops.append(pl.col("mcc_code").n_unique().rolling("event_dttm", period="7d", closed="left").over("customer_id").cast(pl.UInt16).alias("unique_mcc_7d"))
        
    df_lazy = df_lazy.with_columns(rolling_ops)

    if has_session:
         df_lazy = df_lazy.with_columns([
             pl.col("operaton_amt").sum().over(["customer_id", "session_id"]).cast(pl.Float32).alias("session_amt_sum"),
             pl.len().over(["customer_id", "session_id"]).cast(pl.UInt16).alias("session_cnt")
         ])

    # Ratios
    df_lazy = df_lazy.with_columns([
            (pl.col("cnt_1h") / (pl.col("cnt_24h") + 1)).cast(pl.Float32).alias("ratio_cnt_1h_24h"),
            (pl.col("operaton_amt") / (pl.col("amt_24h") + 1)).cast(pl.Float32).alias("ratio_amt_24h"),
            (pl.col("operaton_amt") / (pl.col("avg_amt_7d") + 1)).cast(pl.Float32).alias("ratio_amt_7d"),
            (pl.col("cnt_1h") / (pl.col("cnt_30d") + 1)).cast(pl.Float32).alias("ratio_cnt_1h_30d"),
    ])

    # Фильтрация и сборка
    if is_train:
        print("✂️ Фильтрация Train...")
        start_date = pl.lit("2024-10-01").str.to_datetime()
        
        # 1. Сначала фильтруем по дате (чтобы убрать pre-train) и наличию таргета
        df_lazy = df_lazy.filter(
            (pl.col("event_dttm") >= start_date) & 
            (pl.col("target").is_not_null())
        )
        
        # 2. Адаптивный даунсэмплинг
        # Если DEV_MODE включен, мы УЖЕ взяли 5% юзеров выше.
        # Поэтому тут можно не резать жестко, иначе останется 0 строк.
        if not DEV_MODE:
            df_lazy = df_lazy.filter(
                (pl.col("target") == 1) | ((pl.col("event_id").hash() % 20) == 0)
            )
        else:
            # В DEV_MODE просто берем всё что осталось от выбранных юзеров, 
            # либо делаем мягкий сэмплинг
             df_lazy = df_lazy.filter(
                (pl.col("target") == 1) | ((pl.col("event_id").hash() % 5) == 0) # 20% от 5% юзеров
            )

    else:
        df_lazy = df_lazy.filter(pl.col("is_submit") == 1)

    print("🚀 Сборка (Collect)...")
    df_math = df_lazy.collect()

    print("🔗 Подтягиваем категории...")
    if is_train:
        files_full = sorted(glob.glob(f"{data_path}/train_part_*.parquet"))
    else:
        files_full = sorted([f for f in glob.glob(f"{data_path}/*test*.parquet") if "pre" not in os.path.basename(f)])
    
    if DEV_MODE: files_full = files_full[:2] # Ограничиваем чтение категорий

    try:
        df_cats_lazy = pl.scan_parquet(files_full)
    except:
        df_cats_lazy = pl.scan_parquet(files_full, allow_missing_columns=True)
        
    cols_cats = ["event_id"] + [c for c in CAT_FEATURES if c in df_cats_lazy.collect_schema().names()]

    ids_to_keep = df_math["event_id"]
    df_cats = (
        df_cats_lazy.select(cols_cats)
        .filter(pl.col("event_id").is_in(ids_to_keep))
        .collect()
        .with_columns(pl.col("event_id").cast(pl.Int64))
    )
    
    df_math = df_math.with_columns(pl.col("event_id").cast(pl.Int64))
    df_final = df_math.join(df_cats, on="event_id", how="left")

    for col in CAT_FEATURES:
        if col in df_final.columns:
            df_final = df_final.with_columns(pl.col(col).fill_null("MISSING").cast(pl.String))
        else:
            df_final = df_final.with_columns(pl.lit("MISSING").alias(col))

    print(f"⏱️ Этап занял {time.time()-start_time:.2f} сек. Размер: {df_final.height} строк")
    return df_final


# ==========================================
# 3. АНСАМБЛЬ (5 МОДЕЛЕЙ)
# ==========================================
if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    gc.collect()

    # 1. Готовим данные
    df_train = process_data_stable(DATA_PATH, is_train=True)
    
    if df_train.height == 0:
        raise ValueError("❌ ОШИБКА: Пустой датафрейм после фильтрации! Попробуй выключить DEV_MODE или ослабить фильтры.")

    df_test = process_data_stable(DATA_PATH, is_train=False)

    print("📦 Конвертация теста в Pandas...")
    drop_cols = ["event_id", "customer_id", "event_dttm", "target", "event_desc", "is_submit"]
    features = [c for c in df_train.columns if c not in drop_cols]

    X_test = df_test.select(features).to_pandas()
    for col in CAT_FEATURES:
        if col in X_test.columns: X_test[col] = X_test[col].astype(str)

    final_preds = np.zeros(len(X_test))

    # Для DEV_MODE 1 модель, иначе 5
    SEEDS = [42] if DEV_MODE else [42, 777, 2024, 1337, 555]
    print(f"\n🚀 ЗАПУСК АНСАМБЛЯ ({len(SEEDS)} моделей)...")

    for i, seed in enumerate(SEEDS):
        print(f"\nTraining Model {i + 1}/{len(SEEDS)} [Seed {seed}]...")

        df_sorted = df_train.sort("event_dttm")
        pdf = df_sorted.to_pandas()

        for col in CAT_FEATURES:
            if col in pdf.columns: pdf[col] = pdf[col].astype(str)

        max_date = pdf["event_dttm"].max()
        if max_date is None:
             raise ValueError("❌ Нет данных с датами!")
             
        val_start_date = max_date -  np.timedelta64(42, 'D')
        print(f"📅 Валидация с {val_start_date}")

        mask_train = pdf["event_dttm"] < val_start_date
        mask_val = pdf["event_dttm"] >= val_start_date

        X_train, y_train = pdf[mask_train][features], pdf[mask_train]["target"]
        X_val, y_val = pdf[mask_val][features], pdf[mask_val]["target"]
        
        # Защита от пустого сплита
        if len(X_train) == 0:
            print("⚠️ Пустой Train! Фолбэк на 80/20 по индексу")
            split_idx = int(len(pdf) * 0.8)
            # Еще одна защита, если всего данных < 2 строк
            if split_idx == 0 and len(pdf) > 0: split_idx = 1
            
            X_train, y_train = pdf.iloc[:split_idx][features], pdf.iloc[:split_idx]["target"]
            X_val, y_val = pdf.iloc[split_idx:][features], pdf.iloc[split_idx:]["target"]

        if len(X_train) == 0:
             raise ValueError("❌ КРИТИЧЕСКАЯ ОШИБКА: X_train пуст даже после фолбэка! Проверь входные данные.")

        model = CatBoostClassifier(
            iterations=500 if DEV_MODE else 2000,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=3,
            task_type="CPU", # На ноутбуке всегда CPU безопаснее
            border_count=32,
            max_ctr_complexity=1,
            cat_features=[c for c in CAT_FEATURES if c in features],
            eval_metric="Logloss",
            metric_period=100,
            scale_pos_weight=10,
            early_stopping_rounds=100,
            verbose=100,
            random_seed=seed,
            allow_writing_files=False,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        if not DEV_MODE:
            model.save_model(f"catboost_seed_{seed}.cbm")

        # Если валидация не пустая
        if len(X_val) > 0:
            val_score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
            print(f"🏅 Seed {seed} Local PR-AUC: {val_score:.5f}")

        final_preds += model.predict_proba(X_test)[:, 1]

        del model, pdf, X_train, X_val
        gc.collect()

    final_preds /= len(SEEDS)

    submission = pl.DataFrame({"event_id": df_test["event_id"], "predict": final_preds})
    submission = submission.unique(subset=["event_id"], keep="first")
    submission.write_csv("submission_ensemble_fixed.csv")
    print(f"\n✅ ГОТОВО! Файл: submission_ensemble_fixed.csv")
