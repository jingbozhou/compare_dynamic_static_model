import os, gc
import numpy as np
import pandas as pd

import torch
from tabpfn import TabPFNRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# =========================
# 输入和输出路径
# =========================

OUT_DIR = "/home/zhoujb/project/tmpProject/260811/1-0-2-Rice_18421/"
IN_PATH = os.path.join(OUT_DIR, "trainTestData")

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# TabPFN 单个 region 最大训练样本数
# =========================
# 如果某个 region 的训练样本数超过该值，则随机抽样到该数量。

MAX_TABPFN_TRAIN = 100000
SAMPLE_RANDOM_STATE = 42


# =========================
# 需要分析的性状
# =========================

traits = [
    "Heading date",
    "Culm length",
    "Leaf angle",
    "Leaf width",
    "Panicle number",
    "Leaf length",
    "Panicle length",
    "Plant height"
]


# =========================
# 保存结果
# =========================

results = []


for target_col in traits:

    print("Running trait:", target_col)

    train_file = os.path.join(IN_PATH, f"trainFill-{target_col}.txt")
    test_file = os.path.join(IN_PATH, f"testFill-{target_col}.txt")

    train_df = pd.read_table(train_file)
    test_df = pd.read_table(test_file)

    feature_cols = [c for c in train_df.columns if c.startswith("fea_")]

    # =========
    # 编码 region
    # =========
    # 原代码中编码的是 location 和 year；
    # 当前数据中只使用 region 一个环境变量。

    all_region = pd.concat(
        [train_df["region"], test_df["region"]],
        ignore_index=True
    ).dropna().unique()

    region2id = {v: i for i, v in enumerate(all_region)}

    train_df["region_id"] = train_df["region"].map(region2id)
    test_df["region_id"] = test_df["region"].map(region2id)

    feature_cols = feature_cols + ["region_id"]

    # =========
    # region list
    # =========
    # 按 region 分组循环训练和测试。

    regions = (
        train_df[["region"]]
        .drop_duplicates()
        .sort_values(["region"])
    )

    regions = list(regions["region"])

    # =========
    # TrainingWhole
    # =========
    # 注意：
    # 原始代码中的 TrainingWhole 部分是注释掉的，
    # 因此这里也保留原来的结构，不主动启用。

    X_train_whole = train_df[feature_cols].values
    y_train_whole = train_df[target_col].values

    # model_whole = TabPFNRegressor(
    #     device="cpu",
    #     random_state=0,
    #     n_jobs=-1,
    #     ignore_pretraining_limits=True
    # )

    # model_whole.fit(X_train_whole, y_train_whole)

    # =========
    # 循环 region
    # =========

    for i, region in enumerate(regions):

        test_data = test_df[
            test_df["region"] == region
        ]

        if len(test_data) == 0:
            continue

        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        """
        TrainingWhole
        """

        # preds = model_whole.predict(X_test)

        # whole_r2 = r2_score(y_test, preds)
        # whole_rmse = np.sqrt(mean_squared_error(y_test, preds))
        # whole_pcc = pearsonr(y_test, preds)[0]

        """
        TrainingOnebyOne
        """

        train_data = train_df[
            train_df["region"] == region
        ].copy()

        if len(train_data) < 5:
            continue

        # =========
        # 如果当前 region 的训练样本数超过 10000，则抽样 10000 个
        # =========

        if len(train_data) > MAX_TABPFN_TRAIN:

            print(
                "Sampling training data:",
                "Trait =", target_col,
                "| region =", region,
                "| original train samples =", len(train_data),
                "| used train samples =", MAX_TABPFN_TRAIN
            )

            train_data_use = train_data.sample(
                n=MAX_TABPFN_TRAIN,
                random_state=SAMPLE_RANDOM_STATE
            ).copy()

        else:
            train_data_use = train_data.copy()

        X_train_local = train_data_use[feature_cols].values
        y_train_local = train_data_use[target_col].values

        model_one = TabPFNRegressor(
            device="cuda",
            random_state=0,
            n_preprocessing_jobs=1,
            ignore_pretraining_limits=False
        )

        model_one.fit(X_train_local, y_train_local)

        preds = model_one.predict(X_test)

        one_r2 = r2_score(y_test, preds)
        one_rmse = np.sqrt(mean_squared_error(y_test, preds))
        one_pcc = pearsonr(y_test, preds)[0]

        results.append({

            "Trait": target_col,
            "Round": i + 1,
            "region": region,

            # 原始 region 训练样本数
            "train_local": len(train_data),

            # 实际用于 TabPFN 训练的样本数
            "train_local_used": len(train_data_use),

            "test_data": len(test_data),

            # "TrainingWhole_R2": whole_r2,
            # "TrainingWhole_RMSE": whole_rmse,
            # "TrainingWhole_PCC": whole_pcc,

            "TrainingOnebyOne_R2": one_r2,
            "TrainingOnebyOne_RMSE": one_rmse,
            "TrainingOnebyOne_PCC": one_pcc
        })
        
        # 释放当前 region 的模型和 GPU 显存
        try:
            del model_one
        except NameError:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()


# =========================
# 保存结果
# =========================

result_df = pd.DataFrame(results)

print(result_df)

out_file = os.path.join(OUT_DIR, "tabpfn_compare_results_rice_18421.csv")

result_df.to_csv(out_file, index=False)

print("Results saved to:", out_file)
