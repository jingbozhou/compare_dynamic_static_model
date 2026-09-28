import os
import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


IN_PATH = "/home/zhoujb/project/tmpProject/260811/1-0-1-Rice_3606/trainTestData/"
OUT_FILE = "/home/zhoujb/project/tmpProject/260811/1-0-1-Rice_3606/tabpfn_compare_results_rice_3606.csv"

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

MAX_SAMPLES_V26 = 50000
MAX_FEATURES_V26 = 2000

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check GPU, CUDA, PyTorch, and driver installation.")

DEVICE = "cuda"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

results = []


def calc_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pcc = np.nan
    else:
        pcc = pearsonr(y_true, y_pred)[0]

    return r2, rmse, pcc


def build_tabpfn_v26_model(categorical_features_indices, ignore_pretraining_limits):
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2_6,
        device=DEVICE,
        random_state=0,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=ignore_pretraining_limits,
        n_preprocessing_jobs=1
    )


for target_col in traits:

    print("\nRunning trait:", target_col)

    train_file = os.path.join(IN_PATH, f"trainFill-{target_col}.txt")
    test_file = os.path.join(IN_PATH, f"testFill-{target_col}.txt")

    if not os.path.exists(train_file):
        print(f"Skip {target_col}: train file not found -> {train_file}")
        continue

    if not os.path.exists(test_file):
        print(f"Skip {target_col}: test file not found -> {test_file}")
        continue

    train_df = pd.read_table(train_file)
    test_df = pd.read_table(test_file)

    fea_cols = [c for c in train_df.columns if c.startswith("fea_")]

    # =========
    # 编码 location 和 year
    # =========
    all_loc = pd.concat([train_df["location"], test_df["location"]]).unique()
    all_year = pd.concat([train_df["year"], test_df["year"]]).unique()

    loc2id = {v: i for i, v in enumerate(all_loc)}
    year2id = {v: i for i, v in enumerate(all_year)}

    train_df["location_id"] = train_df["location"].map(loc2id)
    test_df["location_id"] = test_df["location"].map(loc2id)

    train_df["year_id"] = train_df["year"].map(year2id)
    test_df["year_id"] = test_df["year"].map(year2id)

    feature_cols = fea_cols + ["location_id", "year_id"]

    # location_id 和 year_id 是最后两列，明确告诉 TabPFN 这是类别特征
    categorical_features_indices = [
        feature_cols.index("location_id"),
        feature_cols.index("year_id")
    ]

    # =========
    # year-location pairs
    # =========
    pairs = (
        train_df[["year", "location"]]
        .drop_duplicates()
        .sort_values(["year", "location"])
    )

    pairs = list(pairs.itertuples(index=False, name=None))

    # =========
    # TrainingOnebyOne
    # =========
    for i, (year, location) in enumerate(pairs):

        test_data = test_df[
            (test_df["year"] == year) &
            (test_df["location"] == location)
        ]

        if len(test_data) == 0:
            continue

        train_data = train_df[
            (train_df["year"] == year) &
            (train_df["location"] == location)
        ]

        if len(train_data) < 5:
            continue

        X_train_local = train_data[feature_cols].values
        y_train_local = train_data[target_col].values

        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        ignore_pretraining_limits = (
            len(train_data) > MAX_SAMPLES_V26 or
            len(feature_cols) > MAX_FEATURES_V26
        )

        if ignore_pretraining_limits:
            print(
                f"Warning: {target_col} {year}_{location} exceeds TabPFN-2.6 suggested limits "
                f"or feature limits. train_samples={len(train_data)}, features={len(feature_cols)}. "
                f"Using ignore_pretraining_limits=True."
            )

        model_one = build_tabpfn_v26_model(
            categorical_features_indices=categorical_features_indices,
            ignore_pretraining_limits=ignore_pretraining_limits
        )

        model_one.fit(X_train_local, y_train_local)
        preds = model_one.predict(X_test)

        one_r2, one_rmse, one_pcc = calc_metrics(y_test, preds)

        results.append({
            "Trait": target_col,
            "Round": i + 1,
            "year": year,
            "location": location,
            "train_local": len(train_data),
            "test_data": len(test_data),
            "n_features": len(feature_cols),
            "device": DEVICE,
            "model_version": "TabPFN-2.6",
            "ignore_pretraining_limits": ignore_pretraining_limits,
            "TrainingOnebyOne_R2": one_r2,
            "TrainingOnebyOne_RMSE": one_rmse,
            "TrainingOnebyOne_PCC": one_pcc
        })


result_df = pd.DataFrame(results)

print(result_df)

result_df.to_csv(OUT_FILE, index=False)

print("\nAll done.")
print(f"Saved result: {OUT_FILE}")
