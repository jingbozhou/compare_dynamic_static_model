"""
训练 CatBoost 的三种模型：
1. TrainingWhole / FitAll
2. TrainingOnebyOne / FitOne
3. TrainingTwice / Dynamic

本版本修改内容：
1. 将 location/year 双变量类别特征改为 region 单变量类别特征
2. 将环境循环逻辑从 year-location pairs 改为按 region 分组
3. 明确设置 CatBoost 使用 CPU
"""

import os
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# =============================
# 输入和输出路径
# =============================

OUT_DIR = "/data2/zhoujb/project/cropVarCefComb/processData/1-0-3-Cowpea_344/"
IN_PATH = os.path.join(OUT_DIR, "trainTestData")

os.makedirs(OUT_DIR, exist_ok=True)


# =============================
# 性状列表
# =============================

traits = ['Pod length', 'Pod Sugar content', 'Pod Starch content', 'Pod Protein content', 
'Seed Sugar', 'Seed Starch', 'Seed Protein', 'TSW']



# =============================
# CatBoost 参数
# =============================

CATBOOST_PARAMS = {
    "iterations": 1000,
    "thread_count": 48,
    #"task_type": "CPU",
    "verbose": 0
}


# =============================
# 安全计算评价指标
# =============================

def safe_metrics(y_true, preds):
    """
    计算 R2、RMSE 和 PCC。
    如果测试样本数不足，或者 y_true/preds 没有方差，则 PCC 返回 np.nan。
    """

    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    rmse = np.sqrt(mean_squared_error(y_true, preds))

    if len(y_true) >= 2:
        r2 = r2_score(y_true, preds)
    else:
        r2 = np.nan

    if len(y_true) < 2:
        pcc = np.nan
    elif np.std(y_true) == 0:
        pcc = np.nan
    elif np.std(preds) == 0:
        pcc = np.nan
    else:
        pcc = pearsonr(y_true, preds)[0]

    return r2, rmse, pcc


# =============================
# 保存结果
# =============================

results = []


# =============================
# 主循环
# =============================

for target_col in traits:

    print("=" * 80)
    print("Running trait:", target_col)

    train_file = os.path.join(IN_PATH, f"train-{target_col}.txt")
    test_file = os.path.join(IN_PATH, f"test-{target_col}.txt")

    if not os.path.exists(train_file):
        print("Warning: train file does not exist, skip:", train_file)
        continue

    if not os.path.exists(test_file):
        print("Warning: test file does not exist, skip:", test_file)
        continue

    train_df = pd.read_table(train_file)
    test_df = pd.read_table(test_file)

    # =============================
    # 检查必要列
    # =============================

    required_cols = ["region", target_col]

    for col in required_cols:

        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' not found in train file: {train_file}")

        if col not in test_df.columns:
            raise ValueError(f"Column '{col}' not found in test file: {test_file}")

    # =============================
    # region 转换为字符串类型
    # =============================
    # CatBoost 的类别变量建议显式转为 str

    train_df["region"] = train_df["region"].astype(str)
    test_df["region"] = test_df["region"].astype(str)

    # =============================
    # 特征列
    # =============================
    # 原代码中类别变量是 location 和 year；
    # 当前版本只使用 region 一个类别变量。

    feature_cols = [c for c in train_df.columns if c.startswith("fea_")]

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns starting with 'fea_' found for trait: {target_col}")

    cat_features = ["region"]

    feature_cols = feature_cols + cat_features

    # =============================
    # region 分组
    # =============================
    # 原代码中是 year-location pairs；
    # 当前版本改为 region list。

    regions = (
        train_df[["region"]]
        .drop_duplicates()
        .sort_values(["region"])
    )

    regions = list(regions["region"])

    print("Number of features:", len(feature_cols))
    print("Number of categorical features:", len(cat_features))
    print("Number of regions:", len(regions))
    print("Train samples:", len(train_df))
    print("Test samples:", len(test_df))

    # =============================
    # TrainingWhole / FitAll
    # =============================

    whole_pool = Pool(
        data=train_df[feature_cols],
        label=train_df[target_col],
        cat_features=cat_features
    )

    whole_model = CatBoostRegressor(
        **CATBOOST_PARAMS
    )

    whole_model.fit(whole_pool)

    # =============================
    # TrainingTwice 第一阶段
    # =============================
    # first_model 先在全部训练集上训练，
    # 后面每个 region 再基于 first_model 继续训练。

    first_model = CatBoostRegressor(
        **CATBOOST_PARAMS
    )

    first_model.fit(whole_pool)

    # =============================
    # 按 region 测试
    # =============================

    for i, region in enumerate(regions):

        print("-" * 80)
        print("Trait:", target_col, "| region:", region)

        test_data = test_df[
            test_df["region"] == region
        ].copy()

        if len(test_data) == 0:
            print("No test data for this region, skip.")
            continue

        test_pool = Pool(
            data=test_data[feature_cols],
            label=test_data[target_col],
            cat_features=cat_features
        )

        y_true = test_data[target_col].values

        # =============================
        # TrainingWhole / FitAll
        # =============================

        preds = whole_model.predict(test_pool)

        whole_r2, whole_rmse, whole_pcc = safe_metrics(
            y_true,
            preds
        )

        # =============================
        # TrainingOnebyOne / FitOne
        # =============================

        train_data = train_df[
            train_df["region"] == region
        ].copy()

        if len(train_data) == 0:
            print("No train data for this region, skip.")
            continue

        train_pool = Pool(
            data=train_data[feature_cols],
            label=train_data[target_col],
            cat_features=cat_features
        )

        one_model = CatBoostRegressor(
            **CATBOOST_PARAMS
        )

        one_model.fit(train_pool)

        preds = one_model.predict(test_pool)

        one_r2, one_rmse, one_pcc = safe_metrics(
            y_true,
            preds
        )

        # =============================
        # TrainingTwice / Dynamic
        # =============================

        sec_model = CatBoostRegressor(
            **CATBOOST_PARAMS
        )

        sec_model.fit(
            train_pool,
            init_model=first_model
        )

        preds = sec_model.predict(test_pool)

        twice_r2, twice_rmse, twice_pcc = safe_metrics(
            y_true,
            preds
        )

        # =============================
        # 保存结果
        # =============================

        results.append({
            "Trait": target_col,
            "Round": i + 1,
            "region": region,
            "train_data": len(train_data),
            "test_data": len(test_data),
            "feature_number": len(feature_cols),

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

        print(
            "Finished region:",
            region,
            "| Whole PCC:",
            whole_pcc,
            "| One PCC:",
            one_pcc,
            "| Twice PCC:",
            twice_pcc
        )

    # =============================
    # 每个 trait 结束后保存一次中间结果
    # =============================

    result_df = pd.DataFrame(results)

    intermediate_out = os.path.join(
        OUT_DIR,
        "cb_compare_results_cowpea_344.csv"
    )

    result_df.to_csv(
        intermediate_out,
        index=False
    )

    print("Intermediate results saved to:", intermediate_out)


# =============================
# 输出最终结果
# =============================

result_df = pd.DataFrame(results)

print(result_df)

out_file = os.path.join(
    OUT_DIR,
    "cb_compare_results_cowpea_344.csv"
)

result_df.to_csv(
    out_file,
    index=False
)

print("Final results saved to:", out_file)
