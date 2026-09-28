"""
soybean_2k DNN training script (GPU-enabled)

Training strategies are kept consistent with the reference DNN implementation:
1. TrainingWhole / FitAll
2. TrainingOnebyOne / FitOne
3. TrainingTwice / Dynamic

Important:
- soybean_2k keeps separate location and year embeddings because its environments
  are location x year combinations.
- Model architecture, training hyperparameters, and device handling are
  unchanged from the reference implementation.
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
# Input/output path
# =========================

IN_PATH = "/home/zhoujb/project/tmpProject/260811/1-0-4-Soybean_2795/trainTestData/"


# =========================
# Traits
# =========================

traits = [
    "BBD", "BPD", "MD", "FA16C", "FA18C",
    "FAC", "HSW", "SL", "ST", "LL", "LW",
    "NN", "OB", "PLH", "VBN", "PRT",
]


# =========================
# Random seed
# =========================

seed = 120

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# =========================
# Device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

if device == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


# =========================
# DataLoader parameters
# =========================

BATCH_SIZE = 100
PIN_MEMORY = True if device == "cuda" else False
NUM_WORKERS = 0


# =========================
# Dataset
# =========================

class SoybeanDataset(Dataset):

    def __init__(self, df, feat_cols, target):

        self.X = df[feat_cols].values.astype(np.float32)
        self.loc = df["location_id"].values.astype(np.int64)
        self.year = df["year_id"].values.astype(np.int64)
        self.y = df[target].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.loc[idx],
            self.year[idx],
            self.y[idx]
        )


# =========================
# DNN model
# =========================

class SoybeanNet(nn.Module):

    def __init__(self, n_feat, n_loc, n_year):

        super().__init__()

        self.loc_emb = nn.Embedding(n_loc, 2)
        self.year_emb = nn.Embedding(n_year, 2)

        self.mlp = nn.Sequential(
            nn.Linear(n_feat + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, loc, year):

        loc_e = self.loc_emb(loc)
        year_e = self.year_emb(year)

        z = torch.cat([x, loc_e, year_e], dim=1)

        return self.mlp(z).squeeze(1)


# =========================
# Training function
# =========================

def train_model(model, loader, optimizer, device):

    model.train()

    loss_fn = nn.MSELoss()

    for x, loc, year, y in loader:

        x = x.to(device, non_blocking=True)
        loc = loc.to(device, non_blocking=True)
        year = year.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(x, loc, year)

        loss = loss_fn(pred, y)

        loss.backward()

        optimizer.step()


# =========================
# Prediction function
# =========================

def predict(model, loader, device):

    model.eval()

    preds = []
    trues = []

    with torch.no_grad():

        for x, loc, year, y in loader:

            x = x.to(device, non_blocking=True)
            loc = loc.to(device, non_blocking=True)
            year = year.to(device, non_blocking=True)

            pred = model(x, loc, year)

            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    return preds, trues


# =========================
# Safe metrics
# =========================

def safe_metrics(trues, preds):
    """Calculate R2, RMSE and PCC safely."""

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
# DataLoader helper
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
# Results
# =========================

results = []


# =========================
# Main loop
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
    # Required columns
    # =========

    required_cols = ["location", "year", target_col]

    for col in required_cols:
        if col not in train_df.columns:
            raise ValueError(f"Column '{col}' not found in train file: {train_file}")
        if col not in test_df.columns:
            raise ValueError(f"Column '{col}' not found in test file: {test_file}")

    # =========
    # Feature columns
    # =========

    feature_cols = [c for c in train_df.columns if c.startswith("fea_")]

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns starting with 'fea_' found for trait: {target_col}")

    # =========
    # Encode location and year
    # =========

    all_loc = pd.concat(
        [train_df["location"], test_df["location"]],
        ignore_index=True
    ).dropna().unique()

    all_year = pd.concat(
        [train_df["year"], test_df["year"]],
        ignore_index=True
    ).dropna().unique()

    loc2id = {v: i for i, v in enumerate(all_loc)}
    year2id = {v: i for i, v in enumerate(all_year)}

    train_df["location_id"] = train_df["location"].map(loc2id)
    test_df["location_id"] = test_df["location"].map(loc2id)

    train_df["year_id"] = train_df["year"].map(year2id)
    test_df["year_id"] = test_df["year"].map(year2id)

    if train_df["location_id"].isna().any() or test_df["location_id"].isna().any():
        raise ValueError(f"NA found in location_id for trait: {target_col}")

    if train_df["year_id"].isna().any() or test_df["year_id"].isna().any():
        raise ValueError(f"NA found in year_id for trait: {target_col}")

    # =========
    # Environment pairs
    # =========

    pairs = (
        train_df[["year", "location"]]
        .drop_duplicates()
        .sort_values(["year", "location"])
    )

    pairs = list(pairs.itertuples(index=False, name=None))

    print("Number of features:", len(feature_cols))
    print("Number of locations:", len(loc2id))
    print("Number of years:", len(year2id))
    print("Number of environments:", len(pairs))
    print("Number of train samples:", len(train_df))
    print("Number of test samples:", len(test_df))

    # =========
    # TrainingWhole / FitAll
    # =========

    print("TrainingWhole / FitAll ...")

    model_whole = SoybeanNet(
        n_feat=len(feature_cols),
        n_loc=len(loc2id),
        n_year=len(year2id)
    ).to(device)

    optimizer = torch.optim.Adam(
        model_whole.parameters(),
        lr=1e-4
    )

    dataset_train_all = SoybeanDataset(
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
    # TrainingBaseModel for Dynamic
    # =========

    print("TrainingBaseModel for Dynamic ...")

    base_model = SoybeanNet(
        n_feat=len(feature_cols),
        n_loc=len(loc2id),
        n_year=len(year2id)
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
    # Loop over year-location environments
    # =========

    for i, (year, location) in enumerate(pairs):

        print("-" * 80)
        print("Trait:", target_col, "| year:", year, "| location:", location)

        test_data = test_df[
            (test_df["year"] == year) &
            (test_df["location"] == location)
        ].copy()

        if len(test_data) == 0:
            print("No test data for this environment, skip.")
            continue

        train_data = train_df[
            (train_df["year"] == year) &
            (train_df["location"] == location)
        ].copy()

        if len(train_data) == 0:
            print("No train data for this environment, skip.")
            continue

        dataset_test = SoybeanDataset(
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
            "train_data:", len(train_data),
            "| test_data:", len(test_data)
        )

        # =========================
        # TrainingWhole / FitAll test
        # =========================

        preds, trues = predict(
            model_whole,
            test_loader,
            device
        )

        whole_r2, whole_rmse, whole_pcc = safe_metrics(trues, preds)

        # =========================
        # TrainingOnebyOne / FitOne
        # =========================

        print("TrainingOnebyOne / FitOne ...")

        model_one = SoybeanNet(
            n_feat=len(feature_cols),
            n_loc=len(loc2id),
            n_year=len(year2id)
        ).to(device)

        optimizer = torch.optim.Adam(
            model_one.parameters(),
            lr=1e-4
        )

        dataset_train_env = SoybeanDataset(
            train_data,
            feature_cols,
            target_col
        )

        train_loader_env = make_loader(
            dataset_train_env,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(100):
            train_model(
                model_one,
                train_loader_env,
                optimizer,
                device
            )

        preds, trues = predict(
            model_one,
            test_loader,
            device
        )

        one_r2, one_rmse, one_pcc = safe_metrics(trues, preds)

        # =========================
        # TrainingTwice / Dynamic
        # =========================

        print("TrainingTwice / Dynamic ...")

        model_twice = copy.deepcopy(base_model)

        for p in model_twice.parameters():
            p.requires_grad = False

        # Keep the original strategy: fine-tune only the final Linear layer.
        for p in model_twice.mlp[-1].parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model_twice.parameters()),
            lr=1e-5
        )

        dataset_train_env = SoybeanDataset(
            train_data,
            feature_cols,
            target_col
        )

        train_loader_env = make_loader(
            dataset_train_env,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(10):
            train_model(
                model_twice,
                train_loader_env,
                optimizer,
                device
            )

        preds, trues = predict(
            model_twice,
            test_loader,
            device
        )

        twice_r2, twice_rmse, twice_pcc = safe_metrics(trues, preds)

        # =========================
        # Save result
        # =========================

        results.append({
            "Trait": target_col,
            "Round": i + 1,
            "year": year,
            "location": location,
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
            "Finished environment:", year, location,
            "| Whole PCC:", whole_pcc,
            "| One PCC:", one_pcc,
            "| Twice PCC:", twice_pcc
        )

        # Release environment-specific temporary models.
        del model_one
        del model_twice

        if device == "cuda":
            torch.cuda.empty_cache()

        gc.collect()

    # Save intermediate results after each trait.
    result_df = pd.DataFrame(results)

    out_file = os.path.join(
        IN_PATH,
        "../pytorch_compare_results_soybean_2k.csv"
    )

    result_df.to_csv(out_file, index=False)

    print("Intermediate results saved to:", out_file)

    # Release trait-specific global models.
    del model_whole
    del base_model

    if device == "cuda":
        torch.cuda.empty_cache()

    gc.collect()


# =========================
# Final result
# =========================

result_df = pd.DataFrame(results)

print(result_df)

out_file = os.path.join(
    IN_PATH,
    "../pytorch_compare_results_soybean_2k.csv"
)

result_df.to_csv(out_file, index=False)

print("Final results saved to:", out_file)
