# ------------------------------------------------------------
# Блок 1. Импорты, базовые настройки среды и глобальные флаги.
# ------------------------------------------------------------
from pathlib import Path
import os
import gc
import warnings

import numpy as np
import pandas as pd
import polars as pl

from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_cols(200)

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# sampling negatives (green) from train period
NEG_SAMPLE_MOD_RECENT = 10  # from 2025-04-01
NEG_SAMPLE_MOD_OLD = 30     # before 2025-04-01
NEG_SAMPLE_BORDER_STR = "2025-04-01 00:00:00"

# holdout for local validation
VAL_START = pd.Timestamp("2025-05-01")
RECENT_BORDER = pd.Timestamp("2025-02-01")

RANDOM_SEED = 42
FORCE_REBUILD_FEATURES = True
FORCE_REBUILD_PRIORS = False
ADD_CATEGORY_PRIORS = True
USE_GPU = True
RETRAIN_ON_FULL = True

print("DATA_DIR:", DATA_DIR.resolve())
print("CACHE_DIR:", CACHE_DIR.resolve())

# ------------------------------------------------------------
# Блок 2. Списки колонок, фичей и служебных полей.
# ------------------------------------------------------------
BASE_COLS = [
    "customer_id", "event_id", "event_dttm", "event_type_nm", "event_desc",
    "channel_indicator_type", "channel_indicator_sub_type", "operaton_amt", "currency_iso_cd",
    "mcc_code", "pos_cd", "timezone", "session_id", "operating_system_type",
    "battery", "device_system_version", "screen_size", "developer_tools",
    "phone_voip_call_state", "web_rdp_connection", "compromised"
]

FINAL_FEATURE_COLS = [
    # raw (mostly categorical)
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd", "timezone",
    "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
    # event numeric
    "amt", "amt_log_abs", "amt_is_negative", "hour", "weekday", "day", "month",
    "is_weekend", "event_day_number", "battery_pct", "os_ver_major", "screen_w",
    "screen_h", "screen_pixels", "screen_ratio", "session_id",
    # sequence
    "cust_prev_events", "cust_prev_amt_mean", "cust_prev_amt_std", "sec_since_prev_event",
    "amt_delta_prev", "cnt_prev_same_type", "cnt_prev_same_desc", "cnt_prev_same_mcc",
    "cnt_prev_same_subtype", "cnt_prev_same_session", "sec_since_prev_same_type",
    "sec_since_prev_same_desc", "events_before_today",
]

CAT_COLS = [
    "customer_id", "event_type_nm", "event_desc", "channel_indicator_type",
    "channel_indicator_sub_type", "currency_iso_cd", "mcc_code_i", "pos_cd",
    "timezone", "operating_system_type", "phone_voip_call_state", "web_rdp_connection",
    "developer_tools_i", "compromised_i",
]

META_COLS = ["event_id", "period", "event_ts", "is_train_sample", "is_test", "train_target_raw", "target_bin"]

labels_lf = pl.scan_parquet(DATA_DIR / "train_labels.parquet")
labels_df = pl.read_parquet(DATA_DIR / "train_labels.parquet")
print("Labels:", labels_df.shape)

# ------------------------------------------------------------
# Блок 3. Feature engineering: объединение периодов, семплинг
# зеленого класса, временные/поведенческие признаки и target_bin.
# ------------------------------------------------------------
def _period_frames_for_part(part_id: int) -> pl.LazyFrame:
    custs_lf = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .select("customer_id")
        .unique()
    )

    pretrain_lf = (
        pl.scan_parquet(DATA_DIR / f"pretrain_part_{part_id}.parquet")
        .select(BASE_COLS)
        .with_columns(pl.lit("pretrain").alias("period"))
    )
    train_lf = (
        pl.scan_parquet(DATA_DIR / f"train_part_{part_id}.parquet")
        .select(BASE_COLS)
        .with_columns(pl.lit("train").alias("period"))
    )
    pretest_lf = (
        pl.scan_parquet(DATA_DIR / "pretest.parquet")
        .select(BASE_COLS)
        .join(custs_lf, on="customer_id", how="inner")
        .with_columns(pl.lit("pretest").alias("period"))
    )
    test_lf = (
        pl.scan_parquet(DATA_DIR / "test.parquet")
        .select(BASE_COLS)
        .join(custs_lf, on="customer_id", how="inner")
        .with_columns(pl.lit("test").alias("period"))
    )

    return pl.concat([pretrain_lf, train_lf, pretest_lf, test_lf], how="vertical_relaxed")


def build_features_for_part(part_id: int, force: bool = False) -> Path:
    out_path = CACHE_DIR / f"features_part_{part_id}.parquet"
    if out_path.exists() and (not force):
        print(f"[part {part_id}] use cache -> {out_path.name}")
        return out_path

    print(f"[part {part_id}] building features...")
    lf = _period_frames_for_part(part_id)

    # parse and normalize to numeric-only feature space for large-scale windows
    lf = (
        lf.with_columns([
            pl.col("event_dttm").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("event_ts"),
            pl.col("operaton_amt").cast(pl.Float64).alias("amt"),
            pl.col("session_id").cast(pl.Int64, strict=False).fill_null(-1).alias("session_id"),

            pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
            pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
            pl.col("channel_indicator_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_type"),
            pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
            pl.col("currency_iso_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("currency_iso_cd"),
            pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
            pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
            pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
            pl.col("phone_voip_call_state").cast(pl.Int8, strict=False).fill_null(-1).alias("phone_voip_call_state"),
            pl.col("web_rdp_connection").cast(pl.Int8, strict=False).fill_null(-1).alias("web_rdp_connection"),

            pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
            pl.col("battery").str.extract(r"(\\d{1,3})", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("battery_pct"),
            pl.col("device_system_version").str.extract(r"^(\\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("os_ver_major"),
            pl.col("screen_size").str.extract(r"^(\\d+)", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_w"),
            pl.col("screen_size").str.extract(r"x(\\d+)$", 1).cast(pl.Int16, strict=False).fill_null(-1).alias("screen_h"),
            pl.col("developer_tools").cast(pl.Int8, strict=False).fill_null(-1).alias("developer_tools_i"),
            pl.col("compromised").cast(pl.Int8, strict=False).fill_null(-1).alias("compromised_i"),
        ])
        .drop(["event_dttm", "operaton_amt", "mcc_code", "battery", "device_system_version", "screen_size", "developer_tools", "compromised"])
        .sort(["customer_id", "event_ts", "event_id"])
    )

    # labels and sampling mask
    lf = lf.join(labels_lf, on="event_id", how="left")
    lf = lf.with_columns([
        pl.when(pl.col("period") == "train")
          .then(pl.when(pl.col("target").is_null()).then(pl.lit(-1)).otherwise(pl.col("target")))
          .otherwise(pl.lit(None))
          .alias("train_target_raw")
    ])

    border_expr = pl.lit(NEG_SAMPLE_BORDER_STR).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    lf = lf.with_columns([
        ((pl.col("period") == "train") &
         (pl.col("train_target_raw") == -1) &
         (((pl.col("event_ts") >= border_expr) & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED) % NEG_SAMPLE_MOD_RECENT) == 0)) |
          ((pl.col("event_ts") < border_expr) & ((pl.struct(["event_id", "customer_id"]).hash(seed=RANDOM_SEED + 17) % NEG_SAMPLE_MOD_OLD) == 0))))
          .alias("keep_green")
    ])
    lf = lf.with_columns([
        ((pl.col("period") == "train") & ((pl.col("train_target_raw") != -1) | pl.col("keep_green"))).alias("is_train_sample"),
        (pl.col("period") == "test").alias("is_test"),

        pl.col("event_ts").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("event_ts").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("event_ts").dt.day().cast(pl.Int8).alias("day"),
        pl.col("event_ts").dt.month().cast(pl.Int8).alias("month"),
        (pl.col("event_ts").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        (pl.col("event_ts").dt.epoch("s") // 86400).cast(pl.Int32).alias("event_day_number"),
        pl.col("event_ts").dt.date().alias("event_date"),

        pl.col("amt").abs().log1p().cast(pl.Float32).alias("amt_log_abs"),
        (pl.col("amt") < 0).cast(pl.Int8).alias("amt_is_negative"),
        (pl.col("screen_w").cast(pl.Int32) * pl.col("screen_h").cast(pl.Int32)).alias("screen_pixels"),
        pl.when((pl.col("screen_h") > 0) & (pl.col("screen_w") > 0))
          .then(pl.col("screen_w").cast(pl.Float32) / pl.col("screen_h").cast(pl.Float32))
          .otherwise(0.0)
          .alias("screen_ratio"),
    ])

    # sequential customer history features (strictly from previous events after sorting)
    lf = lf.with_columns([
        pl.cum_count("event_id").over("customer_id").cast(pl.Int32).alias("cust_event_idx"),
        pl.col("amt").cum_sum().over("customer_id").alias("cust_cum_amt"),
        (pl.col("amt") * pl.col("amt")).cum_sum().over("customer_id").alias("cust_cum_amt_sq"),
        pl.col("event_ts").shift(1).over("customer_id").alias("prev_event_ts"),
        pl.col("amt").shift(1).over("customer_id").alias("prev_amt"),

        (pl.cum_count("event_id").over(["customer_id", "event_type_nm"]) - 1).cast(pl.Int16).alias("cnt_prev_same_type"),
        (pl.cum_count("event_id").over(["customer_id", "event_desc"]) - 1).cast(pl.Int16).alias("cnt_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "mcc_code_i"]) - 1).cast(pl.Int16).alias("cnt_prev_same_mcc"),
        (pl.cum_count("event_id").over(["customer_id", "channel_indicator_sub_type"]) - 1).cast(pl.Int16).alias("cnt_prev_same_subtype"),
        (pl.cum_count("event_id").over(["customer_id", "session_id"]) - 1).cast(pl.Int16).alias("cnt_prev_same_session"),

        pl.col("event_ts").shift(1).over(["customer_id", "event_type_nm"]).alias("prev_same_type_ts"),
        pl.col("event_ts").shift(1).over(["customer_id", "event_desc"]).alias("prev_same_desc_ts"),
    ])

    lf = lf.with_columns([
        (pl.col("cust_event_idx") - 1).cast(pl.Int32).alias("cust_prev_events"),
        pl.when(pl.col("cust_event_idx") > 1)
          .then((pl.col("cust_cum_amt") - pl.col("amt")) / (pl.col("cust_event_idx") - 1))
          .otherwise(0.0)
          .cast(pl.Float32)
          .alias("cust_prev_amt_mean"),
        pl.when(pl.col("prev_event_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_event_ts")).dt.total_seconds())
          .otherwise(-1)
          .cast(pl.Int32)
          .alias("sec_since_prev_event"),
        (pl.col("amt") - pl.col("prev_amt").fill_null(0.0)).cast(pl.Float32).alias("amt_delta_prev"),
        pl.when(pl.col("prev_same_type_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_type_ts")).dt.total_seconds())
          .otherwise(-1)
          .cast(pl.Int32)
          .alias("sec_since_prev_same_type"),
        pl.when(pl.col("prev_same_desc_ts").is_not_null())
          .then((pl.col("event_ts") - pl.col("prev_same_desc_ts")).dt.total_seconds())
          .otherwise(-1)
          .cast(pl.Int32)
          .alias("sec_since_prev_same_desc"),
        (pl.cum_count("event_id").over(["customer_id", "event_date"]) - 1).cast(pl.Int16).alias("events_before_today"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("cust_event_idx") > 2)
          .then(
              (
                  ((pl.col("cust_cum_amt_sq") - pl.col("amt") * pl.col("amt")) / (pl.col("cust_event_idx") - 1))
                  - (pl.col("cust_prev_amt_mean") * pl.col("cust_prev_amt_mean"))
              )
              .clip(lower_bound=0)
              .sqrt()
          )
          .otherwise(0.0)
          .cast(pl.Float32)
          .alias("cust_prev_amt_std")
    ])

    lf = lf.with_columns([
        pl.when(pl.col("is_train_sample")).then((pl.col("train_target_raw") == 1).cast(pl.Int8)).otherwise(pl.lit(None)).alias("target_bin")
    ])

    select_cols = ["event_id", "period", "event_ts", "is_train_sample", "is_test", "train_target_raw", "target_bin"] + FINAL_FEATURE_COLS

    out_df = (
        lf.filter(pl.col("is_train_sample") | pl.col("is_test"))
          .select(select_cols)
          .collect()
    )

    out_df.write_parquet(out_path, compression="zstd")

    n_train = int(out_df.filter(pl.col("is_train_sample")).height)
    n_test = int(out_df.filter(pl.col("is_test")).height)
    print(f"[part {part_id}] done: rows={out_df.height:,}, train_sample={n_train:,}, test={n_test:,}")

    del out_df
    gc.collect()
    return out_path

# ------------------------------------------------------------
# Блок 4. Сборка и объединение признаков всех трех частей.
# ------------------------------------------------------------
feature_paths = []
for part_id in [1, 2, 3]:
    path = build_features_for_part(part_id, force=FORCE_REBUILD_FEATURES)
    feature_paths.append(path)

features = pl.concat([pl.scan_parquet(p) for p in feature_paths], how="vertical_relaxed").collect()

print("Feature table shape:", features.shape)
print("Train sample rows:", features.filter(pl.col("is_train_sample")).height)
print("Test rows:", features.filter(pl.col("is_test")).height)


# ------------------------------------------------------------
# Блок 5. Category priors (сглаженные статистики по train).
# ------------------------------------------------------------
PRIOR_COL_DEFS = {
    "event_desc": pl.col("event_desc").cast(pl.Int32, strict=False).fill_null(-1).alias("event_desc"),
    "mcc_code_i": pl.col("mcc_code").cast(pl.Int32, strict=False).fill_null(-1).alias("mcc_code_i"),
    "timezone": pl.col("timezone").cast(pl.Int32, strict=False).fill_null(-1).alias("timezone"),
    "operating_system_type": pl.col("operating_system_type").cast(pl.Int16, strict=False).fill_null(-1).alias("operating_system_type"),
    "channel_indicator_sub_type": pl.col("channel_indicator_sub_type").cast(pl.Int16, strict=False).fill_null(-1).alias("channel_indicator_sub_type"),
    "event_type_nm": pl.col("event_type_nm").cast(pl.Int32, strict=False).fill_null(-1).alias("event_type_nm"),
    "pos_cd": pl.col("pos_cd").cast(pl.Int16, strict=False).fill_null(-1).alias("pos_cd"),
}


def _train_scan_with_expr(expr: pl.Expr, key_name: str) -> pl.LazyFrame:
    return pl.concat([
        pl.scan_parquet(DATA_DIR / f"train_part_{i}.parquet")
          .select([pl.col("event_id"), expr])
        for i in [1, 2, 3]
    ], how="vertical_relaxed")


def build_prior_table(key_name: str, expr: pl.Expr, force: bool = False) -> pl.DataFrame:
    out_path = CACHE_DIR / f"prior_{key_name}.parquet"
    if out_path.exists() and (not force):
        return pl.read_parquet(out_path)

    print(f"Building priors for: {key_name}")
    lf = _train_scan_with_expr(expr, key_name)

    cnt_col = f"prior_{key_name}_cnt"
    lbl_cnt_col = f"prior_{key_name}_lbl_cnt"
    red_cnt_col = f"prior_{key_name}_red_cnt"

    total = lf.group_by(key_name).len().rename({"len": cnt_col})
    labeled = (
        lf.join(labels_lf, on="event_id", how="inner")
          .group_by(key_name)
          .agg([
              pl.len().alias(lbl_cnt_col),
              pl.sum("target").cast(pl.Float64).alias(red_cnt_col),
          ])
    )

    prior = (
        total.join(labeled, on=key_name, how="left")
             .with_columns([
                 pl.col(lbl_cnt_col).fill_null(0.0),
                 pl.col(red_cnt_col).fill_null(0.0),
             ])
             .with_columns([
                 ((pl.col(red_cnt_col) + 1.0) / (pl.col(cnt_col) + 200.0)).cast(pl.Float32).alias(f"prior_{key_name}_red_rate_all"),
                 ((pl.col(lbl_cnt_col) + 1.0) / (pl.col(cnt_col) + 200.0)).cast(pl.Float32).alias(f"prior_{key_name}_labeled_rate_all"),
                 ((pl.col(red_cnt_col) + 1.0) / (pl.col(lbl_cnt_col) + 2.0)).cast(pl.Float32).alias(f"prior_{key_name}_red_share_labeled"),
             ])
             .select([
                 key_name,
                 cnt_col,
                 f"prior_{key_name}_red_rate_all",
                 f"prior_{key_name}_labeled_rate_all",
                 f"prior_{key_name}_red_share_labeled",
             ])
             .collect()
    )

    prior.write_parquet(out_path, compression="zstd")
    return prior


prior_feature_cols = []
if ADD_CATEGORY_PRIORS:
    for key_name, expr in PRIOR_COL_DEFS.items():
        prior_df = build_prior_table(key_name, expr, force=FORCE_REBUILD_PRIORS)
        features = features.join(prior_df, on=key_name, how="left")
        prior_feature_cols.extend([c for c in prior_df.columns if c != key_name])

    fill_exprs = [pl.col(c).fill_null(pl.col(c).mean()).alias(c) for c in prior_feature_cols]
    if fill_exprs:
        features = features.with_columns(fill_exprs)

print("Feature table after priors:", features.shape)


# ------------------------------------------------------------
# Блок 6. Подготовка таблиц для CatBoost и контроль leakage.
# ------------------------------------------------------------
train_pl = features.filter(pl.col("is_train_sample")).with_columns([
    pl.col("target_bin").cast(pl.Int8),
])
test_pl = features.filter(pl.col("is_test"))

print("Train sample:", train_pl.shape)
print("Test rows:", test_pl.shape)

train_df = train_pl.to_pandas()
test_df = test_pl.to_pandas()

del features, train_pl, test_pl
gc.collect()

train_df["event_ts"] = pd.to_datetime(train_df["event_ts"])
test_df["event_ts"] = pd.to_datetime(test_df["event_ts"])

feature_cols = [c for c in train_df.columns if c not in META_COLS]
if ADD_CATEGORY_PRIORS:
    feature_cols = [c for c in feature_cols if c != "target"]

# ensure no leakage columns in features
for bad_col in ["target", "keep_green", "event_date"]:
    if bad_col in feature_cols:
        feature_cols.remove(bad_col)

cat_cols = [c for c in CAT_COLS if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]

for c in cat_cols:
    train_df[c] = train_df[c].fillna(-1).astype(np.int64)
    test_df[c] = test_df[c].fillna(-1).astype(np.int64)

# robust fill for numeric columns
medians = train_df[num_cols].median(numeric_only=True)
train_df[num_cols] = train_df[num_cols].fillna(medians)
test_df[num_cols] = test_df[num_cols].fillna(medians)

# keep chronological order for validation split
train_df = train_df.sort_values("event_ts").reset_index(drop=True)

print("Features:", len(feature_cols))
print("Categorical features:", len(cat_cols))
print("Numerical features:", len(num_cols))

Features: 71

# ------------------------------------------------------------
# Блок 7. Функции для обучения CatBoost и рефита на full train.
# ------------------------------------------------------------
def make_weights(raw_target: np.ndarray) -> np.ndarray:
    # raw_target: 1=red, 0=yellow, -1=green sampled
    return np.where(raw_target == 1, 10.0, np.where(raw_target == 0, 2.5, 1.0)).astype(np.float32)


def fit_catboost_with_holdout(X_tr, y_tr, w_tr, X_val, y_val, w_val, cat_cols, params, use_gpu=True):
    params = params.copy()
    params.update({
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": 200,
        "metric_period": 100,
    })

    if use_gpu:
        params.update({"task_type": "GPU", "devices": "0"})
    else:
        params.update({"task_type": "CPU", "thread_count": max(1, (os.cpu_count() or 4) - 1)})

    train_pool = Pool(X_tr, y_tr, weight=w_tr, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, weight=w_val, cat_features=cat_cols)

    try:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    except Exception as e:
        print("GPU fit failed, fallback to CPU:", e)
        params.pop("devices", None)
        params["task_type"] = "CPU"
        params["thread_count"] = max(1, (os.cpu_count() or 4) - 1)
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_raw = model.predict(val_pool, prediction_type="RawFormulaVal")
    val_ap = average_precision_score(y_val, val_raw)
    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = params.get("iterations", 1000)

    print(f"best_iter={best_iter}, val_pr_auc={val_ap:.6f}")
    return model, best_iter, val_ap, params


def refit_full_catboost(X, y, w, cat_cols, base_params, best_iter):
    params = base_params.copy()
    params.pop("od_type", None)
    params.pop("od_wait", None)
    params["iterations"] = int(max(300, best_iter))

    y_arr = np.asarray(y)
    w_arr = np.asarray(w, dtype=np.float32)

    # Safety: guard against scalar or stale/mismatched weights from notebook state
    if w_arr.ndim == 0:
        w_arr = np.full(shape=(len(y_arr),), fill_value=float(w_arr), dtype=np.float32)
    elif w_arr.shape[0] != len(y_arr):
        fill_value = float(np.nanmean(w_arr)) if w_arr.size > 0 else 1.0
        if not np.isfinite(fill_value):
            fill_value = 1.0
        w_arr = np.full(shape=(len(y_arr),), fill_value=fill_value, dtype=np.float32)

    pool = Pool(X, y_arr, weight=w_arr, cat_features=cat_cols)
    model = CatBoostClassifier(**params)
    model.fit(pool, verbose=200)
    return model

# ------------------------------------------------------------
# Блок 8. Обучение MAIN-модели и holdout-валидация.
# ------------------------------------------------------------
val_mask = train_df["event_ts"] >= VAL_START

# MAIN MODEL: full sample timeline
X_main = train_df[feature_cols]
y_main = train_df["target_bin"].astype(np.int8).values
w_main = make_weights(train_df["train_target_raw"].values)

X_main_tr = X_main.loc[~val_mask]
y_main_tr = y_main[~val_mask]
w_main_tr = w_main[~val_mask]

X_main_val = X_main.loc[val_mask]
y_main_val = y_main[val_mask]
w_main_val = w_main[val_mask]

print("Main train rows:", len(X_main_tr), "Main val rows:", len(X_main_val))

params_main = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "od_type": "Iter",
    "od_wait": 300,
}

model_main, best_iter_main, ap_main, used_params_main = fit_catboost_with_holdout(
    X_main_tr, y_main_tr, w_main_tr,
    X_main_val, y_main_val, w_main_val,
    cat_cols=cat_cols,
    params=params_main,
    use_gpu=USE_GPU,
)

# ------------------------------------------------------------
# Блок 9. RECENT-модель и первичный blend main+recent.
# ------------------------------------------------------------
# RECENT MODEL: stronger focus on latest regime + all labeled events
recent_mask = (train_df["event_ts"] >= RECENT_BORDER) | (train_df["train_target_raw"] != -1)
recent_train_mask = recent_mask & (~val_mask)
recent_val_mask = recent_mask & val_mask

X_recent = train_df.loc[recent_mask, feature_cols]
y_recent = train_df.loc[recent_mask, "target_bin"].astype(np.int8).values
w_recent = make_weights(train_df.loc[recent_mask, "train_target_raw"].values)

X_recent_tr = train_df.loc[recent_train_mask, feature_cols]
y_recent_tr = train_df.loc[recent_train_mask, "target_bin"].astype(np.int8).values
w_recent_tr = make_weights(train_df.loc[recent_train_mask, "train_target_raw"].values)

X_recent_val = train_df.loc[recent_val_mask, feature_cols]
y_recent_val = train_df.loc[recent_val_mask, "target_bin"].astype(np.int8).values
w_recent_val = make_weights(train_df.loc[recent_val_mask, "train_target_raw"].values)

print("Recent train rows:", len(X_recent_tr), "Recent val rows:", len(X_recent_val))

params_recent = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "od_type": "Iter",
    "od_wait": 300,
}

model_recent, best_iter_recent, ap_recent, used_params_recent = fit_catboost_with_holdout(
    X_recent_tr, y_recent_tr, w_recent_tr,
    X_recent_val, y_recent_val, w_recent_val,
    cat_cols=cat_cols,
    params=params_recent,
    use_gpu=USE_GPU,
)

# blended holdout score
val_pool_main = Pool(X_main_val, y_main_val, cat_features=cat_cols)
val_pool_recent = Pool(X_recent_val, y_recent_val, cat_features=cat_cols)

pred_main_val = model_main.predict(val_pool_main, prediction_type="RawFormulaVal")
pred_recent_val = model_recent.predict(val_pool_recent, prediction_type="RawFormulaVal")

# align recent val predictions to global val index
val_index = train_df.index[val_mask]
recent_val_index = train_df.index[recent_val_mask]
recent_pred_map = pd.Series(pred_recent_val, index=recent_val_index)
recent_pred_aligned = recent_pred_map.reindex(val_index).fillna(recent_pred_map.mean()).values

blend_val = 0.7 * pred_main_val + 0.3 * recent_pred_aligned
blend_ap = average_precision_score(y_main_val, blend_val)

print(f"Main val PR-AUC:   {ap_main:.6f}")
print(f"Recent val PR-AUC: {ap_recent:.6f}")
print(f"Blend val PR-AUC:  {blend_ap:.6f}")


# ------------------------------------------------------------
# Блок 10. Дополнительные модели, подбор blend и submission.
# ------------------------------------------------------------
# Extra models: suspicious and red|suspicious + better blend
import numpy as np

def _sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))

raw_target = train_df["train_target_raw"].values
X_all = train_df[feature_cols]

# 1) suspicious model: (red + yellow) vs green
y_susp = (raw_target != -1).astype(np.int8)
w_susp = np.where(raw_target != -1, 6.0, 1.2).astype(np.float32)

X_susp_tr = X_all.loc[~val_mask]
y_susp_tr = y_susp[~val_mask]
w_susp_tr = w_susp[~val_mask]

X_susp_val = X_all.loc[val_mask]
y_susp_val = y_susp[val_mask]
w_susp_val = w_susp[val_mask]

params_susp = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "od_type": "Iter",
    "od_wait": 300,
}

model_susp, best_iter_susp, ap_susp, used_params_susp = fit_catboost_with_holdout(
    X_susp_tr, y_susp_tr, w_susp_tr,
    X_susp_val, y_susp_val, w_susp_val,
    cat_cols=cat_cols,
    params=params_susp,
    use_gpu=USE_GPU,
)

# 2) red|suspicious model: red vs yellow on labeled only
labeled_mask = train_df["train_target_raw"].values != -1
labeled_train_mask = labeled_mask & (~val_mask)
labeled_val_mask = labeled_mask & val_mask

y_rg = train_df.loc[labeled_mask, "target_bin"].astype(np.int8).values
w_rg = np.where(train_df.loc[labeled_mask, "train_target_raw"].values == 1, 2.2, 1.0).astype(np.float32)

X_rg_tr = train_df.loc[labeled_train_mask, feature_cols]
y_rg_tr = train_df.loc[labeled_train_mask, "target_bin"].astype(np.int8).values
w_rg_tr = np.where(train_df.loc[labeled_train_mask, "train_target_raw"].values == 1, 2.2, 1.0).astype(np.float32)

X_rg_val = train_df.loc[labeled_val_mask, feature_cols]
y_rg_val = train_df.loc[labeled_val_mask, "target_bin"].astype(np.int8).values
w_rg_val = np.where(train_df.loc[labeled_val_mask, "train_target_raw"].values == 1, 2.2, 1.0).astype(np.float32)

params_rg = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "od_type": "Iter",
    "od_wait": 300,
}

model_rg, best_iter_rg, ap_rg, used_params_rg = fit_catboost_with_holdout(
    X_rg_tr, y_rg_tr, w_rg_tr,
    X_rg_val, y_rg_val, w_rg_val,
    cat_cols=cat_cols,
    params=params_rg,
    use_gpu=USE_GPU,
)

# Holdout predictions (global validation)
val_pool = Pool(X_main_val, y_main_val, cat_features=cat_cols)

pred_main_val = model_main.predict(val_pool, prediction_type="RawFormulaVal")
pred_recent_val = model_recent.predict(val_pool, prediction_type="RawFormulaVal")
pred_susp_val = model_susp.predict(val_pool, prediction_type="RawFormulaVal")
pred_rg_val = model_rg.predict(val_pool, prediction_type="RawFormulaVal")
pred_prod_val = _logit(_sigmoid(pred_susp_val) * _sigmoid(pred_rg_val))

best_ap = -1.0
best_w = None
for blend_w_main in np.arange(0.30, 0.91, 0.05):
    for blend_w_recent in np.arange(0.00, 0.41, 0.05):
        blend_w_prod = 1.0 - blend_w_main - blend_w_recent
        if blend_w_prod < 0:
            continue
        blend = blend_w_main * pred_main_val + blend_w_recent * pred_recent_val + blend_w_prod * pred_prod_val
        ap = average_precision_score(y_main_val, blend)
        if ap > best_ap:
            best_ap = ap
            best_w = (float(blend_w_main), float(blend_w_recent), float(blend_w_prod))

print(f"Main val PR-AUC:   {average_precision_score(y_main_val, pred_main_val):.6f}")
print(f"Recent val PR-AUC: {average_precision_score(y_main_val, pred_recent_val):.6f}")
print(f"Prod val PR-AUC:   {average_precision_score(y_main_val, pred_prod_val):.6f}")
print("Best blend weights (main,recent,prod):", best_w)
print(f"Best blended val PR-AUC: {best_ap:.6f}")

# Refit and test prediction
# Recompute robust weight arrays locally to avoid stale notebook state side-effects
w_main_full = make_weights(train_df["train_target_raw"].values)
w_recent_full = make_weights(train_df.loc[recent_mask, "train_target_raw"].values)

if RETRAIN_ON_FULL:
    print("Refit full MAIN model...")
    model_main_final = refit_full_catboost(
        X_main, y_main, w_main_full,
        cat_cols=cat_cols,
        base_params=used_params_main,
    )

    print("Refit full RECENT model...")
    model_recent_final = refit_full_catboost(
        X_recent, y_recent, w_recent_full,
        cat_cols=cat_cols,
        base_params=used_params_recent,
    )

    print("Refit full SUSP model...")
    model_susp_final = refit_full_catboost(
        X_all, y_susp, w_susp,
        cat_cols=cat_cols,
        base_params=used_params_susp,
    )

    print("Refit full RED|SUSP model...")
    model_rg_final = refit_full_catboost(
        train_df.loc[labeled_mask, feature_cols], y_rg, w_rg,
        cat_cols=cat_cols,
        base_params=used_params_rg,
    )
else:
    model_main_final = model_main
    model_recent_final = model_recent
    model_susp_final = model_susp
    model_rg_final = model_rg

X_test = test_df[feature_cols]
test_pool = Pool(X_test, cat_features=cat_cols)

test_pred_main = model_main_final.predict(test_pool, prediction_type="RawFormulaVal")
test_pred_recent = model_recent_final.predict(test_pool, prediction_type="RawFormulaVal")
test_pred_susp = model_susp_final.predict(test_pool, prediction_type="RawFormulaVal")
test_pred_rg = model_rg_final.predict(test_pool, prediction_type="RawFormulaVal")
test_pred_prod = _logit(_sigmoid(test_pred_susp) * _sigmoid(test_pred_rg))

w_m, w_r, w_p = best_w
test_pred_blend = w_m * test_pred_main + w_r * test_pred_recent + w_p * test_pred_prod

pred_df = pd.DataFrame({
    "event_id": test_df["event_id"].values,
    "predict": test_pred_blend,
})

sample_submit = pd.read_csv(DATA_DIR / "sample_submit.csv")
submission = sample_submit[["event_id"]].merge(pred_df, on="event_id", how="left")

missing = submission["predict"].isna().sum()
print("Submission rows:", len(submission), "Missing predictions:", int(missing))
assert missing == 0, "Some test event_id are missing in predictions"

submission.to_csv("submission.csv", index=False)
print("Saved -> submission.csv")
submission.head()
