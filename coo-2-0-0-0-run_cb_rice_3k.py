"""
训练CatBoost的三种模型，FitAll，FitOne，Dynamic
"""
import os
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

IN_PATH = "/data2/zhoujb/project/cropVarCefComb/processData/1-0-1-Rice_3606/trainTestData/"

traits = [
    "Yield_per_plant",
    "1000-grain_weight",
    "Grain_length_to_width_ratio",
    "Heading_date",
    "Plant_height",
    "Grain_per_panicle",
    "Seed_set_rate",
    "Tiller_number"
]


results = []

for target_col in traits:

    print("Running trait:", target_col)

    train_df = pd.read_table(os.path.join(IN_PATH, f"train-{target_col}.txt"))
    test_df  = pd.read_table(os.path.join(IN_PATH, f"test-{target_col}.txt"))

    train_df["location"] = train_df["location"].astype(str)
    train_df["year"] = train_df["year"].astype(str)

    test_df["location"] = test_df["location"].astype(str)
    test_df["year"] = test_df["year"].astype(str)

    feature_cols = [c for c in train_df.columns if c.startswith("fea_")]
    cat_features = ["location","year"]

    feature_cols = feature_cols + cat_features

    pairs = (
        train_df[["year","location"]]
        .drop_duplicates()
        .sort_values(["year","location"])
    )

    pairs = list(pairs.itertuples(index=False, name=None))

    # =============================
    # TrainingWhole
    # =============================

    whole_pool = Pool(
        data=train_df[feature_cols],
        label=train_df[target_col],
        cat_features=cat_features
    )

    whole_model = CatBoostRegressor(
        iterations=1000,
        thread_count=48,
        verbose=0
    )

    whole_model.fit(whole_pool)

    # =============================
    # TrainingTwice 第一阶段
    # =============================

    first_model = CatBoostRegressor(
        iterations=1000,
        thread_count=48,
        verbose=0
    )

    first_model.fit(whole_pool)

    # =============================
    # 按环境测试
    # =============================

    for i,(year,location) in enumerate(pairs):

        test_data = test_df[
            (test_df["year"]==year) &
            (test_df["location"]==location)
        ]

        if len(test_data) == 0:
            continue

        test_pool = Pool(
            data=test_data[feature_cols],
            label=test_data[target_col],
            cat_features=cat_features
        )

        y_true = test_data[target_col].values

        # =============================
        # TrainingWhole
        # =============================

        preds = whole_model.predict(test_pool)

        whole_r2 = r2_score(y_true, preds)
        whole_rmse = np.sqrt(mean_squared_error(y_true, preds))
        whole_pcc = pearsonr(y_true, preds)[0]

        # =============================
        # TrainingOnebyOne
        # =============================

        train_data = train_df[
            (train_df["year"]==year) &
            (train_df["location"]==location)
        ]

        train_pool = Pool(
            data=train_data[feature_cols],
            label=train_data[target_col],
            cat_features=cat_features
        )

        one_model = CatBoostRegressor(
            iterations=1000,
            thread_count=48,
            verbose=0
        )

        one_model.fit(train_pool)

        preds = one_model.predict(test_pool)

        one_r2 = r2_score(y_true, preds)
        one_rmse = np.sqrt(mean_squared_error(y_true, preds))
        one_pcc = pearsonr(y_true, preds)[0]

        # =============================
        # TrainingTwice
        # =============================

        sec_model = CatBoostRegressor(
            iterations=1000,
            thread_count=48,
            verbose=0
        )

        sec_model.fit(train_pool, init_model=first_model)

        preds = sec_model.predict(test_pool)

        twice_r2 = r2_score(y_true, preds)
        twice_rmse = np.sqrt(mean_squared_error(y_true, preds))
        twice_pcc = pearsonr(y_true, preds)[0]

        results.append({
            "Trait": target_col,
            "Round": i+1,
            "year": year,
            "location": location,
            "test_data": len(test_data),

            "TraingWhole_R2": whole_r2,
            "TraingWhole_RMSE": whole_rmse,
            "TraingWhole_PCC": whole_pcc,

            "TraingOnebyOne_R2": one_r2,
            "TraingOnebyOne_RMSE": one_rmse,
            "TraingOnebyOne_PCC": one_pcc,

            "TraingTwice_R2": twice_r2,
            "TraingTwice_RMSE": twice_rmse,
            "TraingTwice_PCC": twice_pcc
        })


# =============================
# 输出结果
# =============================

result_df = pd.DataFrame(results)

print(result_df)


result_df.to_csv(os.path.join(IN_PATH, "cb_compare_results_rice_3606.csv"), index=False)