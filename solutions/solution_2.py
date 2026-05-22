import polars as pl
import numpy as np
import gc
import glob
import os
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

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

ESSENTIAL_COLS = [
    "event_id",
    "customer_id",
    "event_dttm",
    "operaton_amt",
    "mcc_code",
    "target",
]


# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (ТОТ ЖЕ C РОТАЦИЕЙ)
# ==========================================
def process_data_iterative(data_path, is_train=True, seed_offset=0):
    mode = "TRAIN" if is_train else "TEST"
    # ... (Тут сокращу для краткости, используй ТУ ЖЕ функцию process_data_iterative из прошлого ответа) ...
    # ... (Она у тебя уже есть, просто скопируй её сюда из предыдущего моего сообщения) ...
    # ВСТАВЬ СЮДА ПОЛНЫЙ КОД process_data_iterative
    # Если нужно, я продублирую, но она идентична той, что была выше.

    # --- ДУБЛЬ ФУНКЦИИ ДЛЯ УДОБСТВА ---
    if is_train:
        files = sorted(glob.glob(f"{data_path}/train_part_*.parquet"))
        df_lazy = pl.scan_parquet(files)
        labels = pl.scan_parquet(f"{data_path}/train_labels.parquet").select(
            ["event_id", "target"]
        )
        df_lazy = df_lazy.join(labels, on="event_id", how="left").with_columns(
            pl.col("target").fill_null(0)
        )

        # Rotating Downsampling
        df_lazy = df_lazy.filter(
            (pl.col("target") == 1)
            | (((pl.col("event_id").hash() + seed_offset) % 100) < 5)
        )
        df_lazy = df_lazy.filter(pl.col("event_dttm") >= pl.lit("2024-10-01"))
        cols = [c for c in ESSENTIAL_COLS if c in df_lazy.columns]
        df_lazy = df_lazy.select(cols)
    else:
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

    df_lazy = df_lazy.with_columns(pl.col("event_dttm").str.to_datetime())
    df_lazy = df_lazy.sort(["customer_id", "event_dttm"])

    df_lazy = df_lazy.with_columns(
        [
            pl.col("event_dttm").dt.hour().alias("hour"),
            pl.col("event_dttm").dt.weekday().alias("day_of_week"),
            (pl.col("event_dttm").dt.hour() < 6).cast(pl.Int8).alias("is_night"),
            pl.col("event_dttm")
            .diff()
            .dt.total_seconds()
            .over("customer_id")
            .fill_null(999999)
            .cast(pl.Float32)
            .alias("seconds_since_prev"),
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
            .alias("avg_amt_7d"),
            pl.len()
            .rolling("event_dttm", period="24h", closed="left")
            .over(["customer_id", "mcc_code"])
            .cast(pl.UInt16)
            .alias("mcc_count_24h"),
        ]
    )

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

    if not is_train:
        df_lazy = df_lazy.filter(pl.col("is_submit") == 1)

    df_math = df_lazy.collect()

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
                pl.col(col).fill_null("MISSING").cast(pl.Categorical)
            )

    return df_final


# ==========================================
# 3. STACKING (КОНВЕЙЕР МОНСТРОВ)
# ==========================================
if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    gc.collect()

    # 1. Готовим TEST (один раз)
    df_test = process_data_iterative(DATA_PATH, is_train=False)
    features = [
        c
        for c in df_test.columns
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

    # Данные для моделей
    X_test_pd = df_test.select(features).to_pandas()
    for col in CAT_FEATURES:
        if col in X_test_pd.columns:
            X_test_pd[col] = X_test_pd[col].astype("category")

    # Сюда будем складывать предсказания от КАЖДОГО фолда
    # Чтобы потом обучить Босса (Meta-model)
    meta_X_train = []  # Предсказания на валидации (для обучения босса)
    meta_y_train = []  # Реальные ответы на валидации

    meta_X_test_cat = np.zeros(len(X_test_pd))  # Предсказания CatBoost на тесте
    meta_X_test_lgbm = np.zeros(len(X_test_pd))  # Предсказания LightGBM на тесте

    SEEDS = [42, 777, 2024, 1337, 555]

    print(f"\n🚀 ЗАПУСК STACKING ({len(SEEDS)} фолдов)...")

    for i, seed in enumerate(SEEDS):
        print(f"\nTraining Fold {i + 1}/{len(SEEDS)} [Seed {seed}]...")

        # 1. Грузим свежий кусок данных
        df_train = process_data_iterative(DATA_PATH, is_train=True, seed_offset=i * 13)
        df_sorted = df_train.sort("event_dttm")
        pdf = df_sorted.to_pandas()

        for col in CAT_FEATURES:
            if col in pdf.columns:
                pdf[col] = pdf[col].astype("category")

        # 2. Time Split (Окт-Март = Трейн, Апр-Май = Валидация)
        split_date = "2025-04-01"
        mask_train = pdf["event_dttm"] < split_date
        mask_val = pdf["event_dttm"] >= split_date

        X_train, y_train = pdf[mask_train][features], pdf[mask_train]["target"]
        X_val, y_val = pdf[mask_val][features], pdf[mask_val]["target"]

        # === LEVEL 1: CatBoost ===
        print("   🐱 CatBoost...")
        cb = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=3,
            scale_pos_weight=5,
            task_type="GPU",
            devices="0",
            cat_features=[c for c in CAT_FEATURES if c in features],
            eval_metric="Logloss",
            early_stopping_rounds=50,
            verbose=0,
            random_seed=seed,
            allow_writing_files=False,
        )
        cb.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

        # Предсказываем на Валидации (для Босса) и на Тесте (для финала)
        val_pred_cat = cb.predict_proba(X_val)[:, 1]
        test_pred_cat = cb.predict_proba(X_test_pd)[:, 1]

        # === LEVEL 1: LightGBM ===
        print("   💡 LightGBM...")
        lgbm = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.06,
            num_leaves=31,
            scale_pos_weight=5,
            metric="binary_logloss",
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        # Callbacks для LGBM
        from lightgbm import early_stopping, log_evaluation

        lgbm.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[early_stopping(50), log_evaluation(0)],
        )

        # Если хочешь убрать ошибку в редакторе (не обязательно)
        val_pred_lgbm = np.array(lgbm.predict_proba(X_val))[:, 1]
        test_pred_lgbm = np.array(lgbm.predict_proba(X_test_pd))[:, 1]

        # === СОБИРАЕМ ДАННЫЕ ДЛЯ БОССА ===
        # Босс увидит: [Предсказание_CatBoost, Предсказание_LGBM]
        fold_meta_features = np.column_stack((val_pred_cat, val_pred_lgbm))
        meta_X_train.append(fold_meta_features)
        meta_y_train.append(y_val)

        # Накапливаем предсказания на тесте
        meta_X_test_cat += test_pred_cat
        meta_X_test_lgbm += test_pred_lgbm

        score_cb = average_precision_score(y_val, val_pred_cat)
        score_lgbm = average_precision_score(y_val, val_pred_lgbm)
        print(f"   🏅 Scores -> Cat: {score_cb:.4f} | LGBM: {score_lgbm:.4f}")

        del cb, lgbm, pdf, X_train, X_val, df_train
        gc.collect()

    # === LEVEL 2: ФИНАЛИЗАЦИЯ (META-MODEL) ===
    print("\n🧠 Обучаем Босса (Logistic Regression)...")

    # Объединяем все валидационные предсказания в одну большую кучу
    full_meta_X = np.vstack(meta_X_train)
    full_meta_y = np.concatenate(meta_y_train)

    # Босс учится находить баланс
    meta_model = LogisticRegression()
    meta_model.fit(full_meta_X, full_meta_y)

    print(
        f"⚖️ Веса Босса: CatBoost={meta_model.coef_[0][0]:.2f}, LightGBM={meta_model.coef_[0][1]:.2f}"
    )

    # Готовим итоговые данные для теста
    # Сначала усредняем накопленные предсказания
    avg_test_cat = meta_X_test_cat / len(SEEDS)
    avg_test_lgbm = meta_X_test_lgbm / len(SEEDS)

    # Формируем вход для Босса
    test_meta_features = np.column_stack((avg_test_cat, avg_test_lgbm))

    # БОСС ДЕЛАЕТ ФИНАЛЬНЫЙ ВЕРДИКТ
    final_preds = meta_model.predict_proba(test_meta_features)[:, 1]

    submission = pl.DataFrame({"event_id": df_test["event_id"], "predict": final_preds})
    submission = submission.unique(subset=["event_id"], keep="first")
    submission.write_csv("submission_stacking.csv")
    print(f"\n✅ ГОТОВО! Файл: submission_stacking.csv")
