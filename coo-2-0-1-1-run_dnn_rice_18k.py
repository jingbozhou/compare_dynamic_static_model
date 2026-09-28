"""
训练 DNN 的三种模型：
1. TrainingWhole / FitAll
2. TrainingOnebyOne / FitOne
3. TrainingTwice / Dynamic

本版本修改内容：
1. 将 location/year 双变量编码改为 region 单变量编码
2. 将环境循环逻辑从 year-location pairs 改为按 region 分组
3. 保留原有三种训练策略
4. 增强 GPU 使用逻辑
"""

import os
import gc
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr


# =========================
# 输入和输出路径
# =========================

OUT_DIR = "/home/zhoujb/project/tmpProject/260811/1-0-2-Rice_18421/"
IN_PATH = os.path.join(OUT_DIR, "trainTestData")

os.makedirs(OUT_DIR, exist_ok=True)


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
# 随机种子
# =========================

seed = 120

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# =========================
# 设备设置
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


# =========================
# DataLoader 参数
# =========================

BATCH_SIZE = 100
PIN_MEMORY = True if device == "cuda" else False
NUM_WORKERS = 0


# =========================
# Dataset
# =========================

class RiceDataset(Dataset):

    def __init__(self, df, feat_cols, target):

        self.X = df[feat_cols].values.astype(np.float32)

        # 原代码使用 location_id 和 year_id
        # 当前版本改为只使用 region_id
        self.region = df["region_id"].values.astype(np.int64)

        self.y = df[target].values.astype(np.float32)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.region[idx],
            self.y[idx]
        )


# =========================
# DNN 模型
# =========================

class RiceNet(nn.Module):

    def __init__(self, n_feat, n_region):

        super().__init__()

        # 原代码：
        # self.loc_emb = nn.Embedding(n_loc, 2)
        # self.year_emb = nn.Embedding(n_year, 2)
        #
        # 当前版本只使用 region embedding
        self.region_emb = nn.Embedding(n_region, 2)

        self.mlp = nn.Sequential(
            nn.Linear(n_feat + 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, region):

        region_e = self.region_emb(region)

        z = torch.cat([x, region_e], dim=1)

        return self.mlp(z).squeeze(1)


# =========================
# 训练函数
# =========================

def train_model(model, loader, optimizer, device):

    model.train()

    loss_fn = nn.MSELoss()

    for x, region, y in loader:

        x = x.to(device, non_blocking=True)
        region = region.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(x, region)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()


# =========================
# 预测函数
# =========================

def predict(model, loader, device):

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for x, region, y in loader:

            x = x.to(device, non_blocking=True)
            region = region.to(device, non_blocking=True)

            pred = model(x, region)

            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return preds, trues


# =========================
# 安全计算评价指标
# =========================

def safe_metrics(trues, preds):
    """
    计算 R2、RMSE、PCC。
    如果测试样本数不足，或 trues/preds 没有方差，则 PCC 返回 np.nan。
    """

    trues = np.asarray(trues)
    preds = np.asarray(preds)

    if len(trues) == 0:
        return np.nan, np.nan, np.nan

    if len(trues) >= 2:
        r2 = r2_score(trues, preds)
    else:
        r2 = np.nan

    rmse = np.sqrt(mean_squared_error(trues, preds))

    if len(trues) < 2:
        pcc = np.nan
    elif np.std(trues) == 0:
        pcc = np.nan
    elif np.std(preds) == 0:
        pcc = np.nan
    else:
        pcc = pearsonr(trues, preds)[0]

    return r2, rmse, pcc


# =========================
# 创建 DataLoader
# =========================

def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


# =========================
# 保存结果
# =========================

results = []


# =========================
# 主循环
# =========================

for target_col in traits:

    print("=" * 80)
    print("Running trait:", target_col)

    train_file = os.path.join(IN_PATH, f"trainFill-{target_col}.txt")
    test_file = os.path.join(IN_PATH, f"testFill-{target_col}.txt")

    if not os.path.exists(train_file):
        print("Warning: train file does not exist, skip:", train_file)
        continue

    if not os.path.exists(test_file):
        print("Warning: test file does not exist, skip:", test_file)
        continue

    train_df = pd.read_table(train_file)
    test_df = pd.read_table(test_file)

    # =========
    # 检查必要列
    # =========

    required_cols = ["region", target_col]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' not found in train file: {train_file}")
        if col not in test_df.columns:
            raise ValueError(f"Column '{col}' not found in test file: {test_file}")

    # =========
    # 特征列
    # =========

    feature_cols = [c for c in train_df.columns if c.startswith("fea_")]

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns starting with 'fea_' found for trait: {target_col}")

    # =========
    # 编码 region
    # =========
    # 原代码中编码 location 和 year；
    # 当前版本只编码 region。

    all_region = pd.concat(
        [train_df["region"], test_df["region"]],
        ignore_index=True
    ).dropna().unique()

    region2id = {v: i for i, v in enumerate(all_region)}

    train_df["region_id"] = train_df["region"].map(region2id)
    test_df["region_id"] = test_df["region"].map(region2id)

    if train_df["region_id"].isna().any():
        raise ValueError(f"NA found in train_df['region_id'] for trait: {target_col}")

    if test_df["region_id"].isna().any():
        raise ValueError(f"NA found in test_df['region_id'] for trait: {target_col}")

    n_region = len(region2id)

    # =========
    # region list
    # =========
    # 原代码中是 year-location pairs；
    # 当前版本改为按 region 分组。

    regions = (
        train_df[["region"]]
        .drop_duplicates()
        .sort_values(["region"])
    )

    regions = list(regions["region"])

    print("Number of features:", len(feature_cols))
    print("Number of regions:", n_region)
    print("Number of train samples:", len(train_df))
    print("Number of test samples:", len(test_df))

    # =========
    # TrainingWhole / FitAll
    # =========

    print("TrainingWhole / FitAll ...")

    model_whole = RiceNet(
        n_feat=len(feature_cols),
        n_region=n_region
    ).to(device)

    optimizer = torch.optim.Adam(
        model_whole.parameters(),
        lr=1e-4
    )

    dataset_train_all = RiceDataset(
        train_df,
        feature_cols,
        target_col
    )

    train_loader_all = make_loader(
        dataset_train_all,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for epoch in range(100):
        train_model(
            model_whole,
            train_loader_all,
            optimizer,
            device
        )

    # =========
    # TrainingBaseModel
    # =========
    # base_model 用于后面的 TrainingTwice / Dynamic。
    # 这里保留原代码逻辑：base_model 也是先在全训练集上训练 100 epoch。

    print("TrainingBaseModel for Dynamic ...")

    base_model = RiceNet(
        n_feat=len(feature_cols),
        n_region=n_region
    ).to(device)

    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=1e-4
    )

    for epoch in range(100):
        train_model(
            base_model,
            train_loader_all,
            optimizer,
            device
        )

    # =========
    # 循环 region
    # =========

    for i, region in enumerate(regions):

        print("-" * 80)
        print("Trait:", target_col, "| region:", region)

        test_data = test_df[
            test_df["region"] == region
        ].copy()

        if len(test_data) == 0:
            print("No test data for this region, skip.")
            continue

        train_data = train_df[
            train_df["region"] == region
        ].copy()

        if len(train_data) == 0:
            print("No train data for this region, skip.")
            continue

        dataset_test = RiceDataset(
            test_data,
            feature_cols,
            target_col
        )

        test_loader = make_loader(
            dataset_test,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        print(
            "train_data:",
            len(train_data),
            "| test_data:",
            len(test_data)
        )

        # =========================
        # TrainingWhole / FitAll 测试
        # =========================

        preds, trues = predict(
            model_whole,
            test_loader,
            device
        )

        whole_r2, whole_rmse, whole_pcc = safe_metrics(
            trues,
            preds
        )

        # =========================
        # TrainingOnebyOne / FitOne
        # =========================

        print("TrainingOnebyOne / FitOne ...")

        model_one = RiceNet(
            n_feat=len(feature_cols),
            n_region=n_region
        ).to(device)

        optimizer = torch.optim.Adam(
            model_one.parameters(),
            lr=1e-4
        )

        dataset_train_region = RiceDataset(
            train_data,
            feature_cols,
            target_col
        )

        train_loader_region = make_loader(
            dataset_train_region,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(100):
            train_model(
                model_one,
                train_loader_region,
                optimizer,
                device
            )

        preds, trues = predict(
            model_one,
            test_loader,
            device
        )

        one_r2, one_rmse, one_pcc = safe_metrics(
            trues,
            preds
        )

        # =========================
        # TrainingTwice / Dynamic
        # =========================

        print("TrainingTwice / Dynamic ...")

        model_twice = copy.deepcopy(base_model)

        # 冻结所有参数
        for p in model_twice.parameters():
            p.requires_grad = False

        # 只微调 MLP 最后一层
        for p in model_twice.mlp[-1].parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_twice.parameters()),
            lr=1e-5
        )

        dataset_train_region = RiceDataset(
            train_data,
            feature_cols,
            target_col
        )

        train_loader_region = make_loader(
            dataset_train_region,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(10):
            train_model(
                model_twice,
                train_loader_region,
                optimizer,
                device
            )

        preds, trues = predict(
            model_twice,
            test_loader,
            device
        )

        twice_r2, twice_rmse, twice_pcc = safe_metrics(
            trues,
            preds
        )

        # =========================
        # 保存结果
        # =========================

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

        # 释放当前 region 的临时模型
        del model_one
        del model_twice

        if device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    # 每个 trait 跑完后保存一次中间结果，避免中途报错导致全部结果丢失
    result_df = pd.DataFrame(results)

    out_file = os.path.join(
        OUT_DIR,
        "pytorch_compare_results_rice_18421.csv"
    )

    result_df.to_csv(
        out_file,
        index=False
    )

    print("Intermediate results saved to:", out_file)

    # 释放当前 trait 的全局模型
    del model_whole
    del base_model

    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()


# =========================
# 保存最终结果
# =========================

result_df = pd.DataFrame(results)

print(result_df)

out_file = os.path.join(
    OUT_DIR,
    "pytorch_compare_results_rice_18421.csv"
)

result_df.to_csv(
    out_file,
    index=False
)

print("Final results saved to:", out_file)
