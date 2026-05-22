import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. СПИСОК ФАЙЛОВ ДЛЯ АНАЛИЗА
# ==========================================
files = [
    "submission_ensemble_1.csv",
    "submission_ensemble_8.vcsv",
]

# Проверим, какие файлы реально существуют
existing_files = [f for f in files if os.path.exists(f)]
if not existing_files:
    print("❌ Нет ни одного файла с предсказаниями. Проверьте имена файлов.")
    exit()

print(f"Найдено файлов: {len(existing_files)}")
for f in existing_files:
    print(f"  - {f}")

# ==========================================
# 2. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ
# ==========================================
dfs = []
for f in existing_files:
    df = pl.read_csv(f).select(["event_id", "predict"])
    # Короткое имя (убираем "submission_" и ".csv")
    short_name = f.replace("submission_", "").replace(".csv", "")
    df = df.rename({"predict": f"pred_{short_name}"})
    dfs.append(df)

# Объединяем все по event_id (inner join – оставляем только общие event_id)
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.join(df, on="event_id", how="inner")

print(f"\nВсего операций после объединения: {merged.height}")

# ==========================================
# 3. СТАТИСТИКИ ПО МОДЕЛЯМ
# ==========================================
pred_cols = [c for c in merged.columns if c.startswith("pred_")]

merged = merged.with_columns(
    [
        pl.mean_horizontal(pred_cols).alias("mean_pred"),
        pl.concat_list(pred_cols).list.std().alias("std_pred"),
    ]
)

# Самые спорные (наибольшее std)
controversial = merged.sort("std_pred", descending=True)

# Сохраняем для ручного анализа
out_cols = ["event_id", "mean_pred", "std_pred"] + pred_cols
controversial.select(out_cols).write_csv("controversial_predictions.csv")
print(
    "\n✅ Сохранён файл 'controversial_predictions.csv' с топ-операциями по разногласиям."
)

# ==========================================
# 4. ВЫВОД ТОП-10 СПОРНЫХ
# ==========================================
print("\n🔍 Топ-10 самых спорных операций (наибольшее std):")
print(controversial.head(10).select(["event_id", "mean_pred", "std_pred"] + pred_cols))

# ==========================================
# 5. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# ==========================================
print("\n📊 Корреляционная матрица между моделями:")
corr_matrix = merged.select(pred_cols).to_pandas().corr()
print(corr_matrix)

# Визуализация корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Корреляции между предсказаниями моделей")
plt.tight_layout()
plt.savefig("model_correlations.png", dpi=150)
print("📈 График корреляций сохранён как 'model_correlations.png'")

# ==========================================
# 6. РАСПРЕДЕЛЕНИЯ
# ==========================================
# Гистограмма средних предсказаний
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(
    merged["mean_pred"].to_numpy(),
    bins=50,
    alpha=0.7,
    color="steelblue",
    edgecolor="black",
)
plt.title("Распределение средних предсказаний")
plt.xlabel("Средняя вероятность")
plt.ylabel("Частота")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Гистограмма стандартных отклонений (разногласия)
plt.subplot(1, 2, 2)
plt.hist(
    merged["std_pred"].to_numpy(), bins=50, alpha=0.7, color="coral", edgecolor="black"
)
plt.title("Распределение разногласий (std)")
plt.xlabel("Стандартное отклонение")
plt.ylabel("Частота")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("predictions_distributions.png", dpi=150)
print("📊 Распределения сохранены в 'predictions_distributions.png'")

plt.show()  # если нужно увидеть на экране
