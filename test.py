import polars as pl

labels = pl.read_parquet("./data/train_labels.parquet")
print(labels["target"].value_counts())
# Если там только 0 и 1, значит Желтые помечены как 0.
# Тогда ищи их в самих данных (train_part) по каким-то флагам типа "verificatioprocess_data_stablee.

# Загрузи старый лучший и новый
old = pl.read_csv("submission_ensemble_2.csv")  # твой на 0.093
new = pl.read_csv("submission_ensemble_6.csv")

# Соедини их по ID
comp = old.join(new, on="event_id", suffix="_new")

# Посмотри корреляцию
corr = comp.select(pl.corr("predict", "predict_new")).item()
print(f"Сходство моделей: {corr:.4f}")
