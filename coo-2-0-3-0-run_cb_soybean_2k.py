"""Evaluate CatBoost FitAll, FitOne and FT on the soybean_2k data.

The default paths target the server copy of the pre-split Soybean_2795 data.
Models are trained independently for each trait. FitAll uses all training
environments, FitOne uses one location-year environment, and FT continues a
separately fitted global model with the corresponding local training records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


DEFAULT_INPUT_DIR = Path(
    "/data2/zhoujb/project/cropVarCefComb/processData/"
    "1-0-4-Soybean_2795/trainTestData"
)
DEFAULT_OUTPUT_NAME = "cb_compare_results_soybean_2k.csv"

TRAITS = [
    "BBD", "BPD", "MD", "FA16C", "FA18C",
    "FAC", "HSW", "SL", "ST", "LL", "LW",
    "NN", "OB", "PLH", "VBN", "PRT",
]

ENV_COLS = ["year", "location"]
CAT_FEATURES = ["location", "year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run soybean_2k CatBoost FitAll, FitOne and FT evaluation."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path. By default, the result is written inside "
            "--input-dir as cb_compare_results_soybean_2k.csv."
        ),
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all input files without importing CatBoost or training models.",
    )
    return parser.parse_args()


def safe_metrics(
    y_true: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, float]:
    """Return R2, RMSE and PCC, using NaN when a metric is undefined."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    r2 = float(r2_score(y_true, predictions)) if len(y_true) >= 2 else np.nan

    if (
        len(y_true) < 2
        or np.std(y_true) == 0
        or np.std(predictions) == 0
    ):
        pcc = np.nan
    else:
        pcc = float(pearsonr(y_true, predictions)[0])

    return r2, rmse, pcc


def read_trait_data(
    input_dir: Path, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[tuple[str, str]]]:
    """Read and validate one trait's pre-split train and test tables."""
    train_file = input_dir / f"train-{target_col}.txt"
    test_file = input_dir / f"test-{target_col}.txt"
    if not train_file.is_file():
        raise FileNotFoundError(train_file)
    if not test_file.is_file():
        raise FileNotFoundError(test_file)

    train_df = pd.read_table(train_file)
    test_df = pd.read_table(test_file)
    required_cols = {"ID", "location", "year", target_col}
    for path, frame in ((train_file, train_df), (test_file, test_df)):
        missing = required_cols.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        if frame[list(required_cols)].isna().any().any():
            raise ValueError(f"{path.name} contains missing IDs, environments or labels")
        if frame.duplicated(["ID", "location", "year"]).any():
            raise ValueError(f"{path.name} contains duplicate ID-location-year rows")

        frame["location"] = frame["location"].astype(str)
        frame["year"] = frame["year"].astype(str)

    numeric_features = [c for c in train_df.columns if c.startswith("fea_")]
    if not numeric_features:
        raise ValueError(f"No fea_ columns found for trait: {target_col}")
    test_features = [c for c in test_df.columns if c.startswith("fea_")]
    if numeric_features != test_features:
        raise ValueError(
            f"Train/test fea_ columns differ for trait: {target_col}"
        )

    train_environments = set(
        map(tuple, train_df[ENV_COLS].drop_duplicates().to_numpy())
    )
    test_environments = set(
        map(tuple, test_df[ENV_COLS].drop_duplicates().to_numpy())
    )
    if train_environments != test_environments:
        train_only = sorted(train_environments - test_environments)
        test_only = sorted(test_environments - train_environments)
        raise ValueError(
            f"Train/test environments differ for {target_col}; "
            f"train_only={train_only}, test_only={test_only}"
        )

    environments = sorted(train_environments)
    feature_cols = [*numeric_features, *CAT_FEATURES]
    return train_df, test_df, feature_cols, environments


def make_model(args: argparse.Namespace):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=args.iterations,
        thread_count=args.threads,
        random_seed=args.random_seed,
        verbose=0,
    )


def run_experiment(args: argparse.Namespace, input_dir: Path) -> pd.DataFrame:
    from catboost import Pool

    results: list[dict[str, object]] = []

    for target_col in TRAITS:
        print("=" * 80)
        print("Running trait:", target_col)
        train_df, test_df, feature_cols, environments = read_trait_data(
            input_dir, target_col
        )
        print("Number of features:", len(feature_cols))
        print("Number of environments:", len(environments))
        print("Train samples:", len(train_df), "| Test samples:", len(test_df))

        whole_pool = Pool(
            data=train_df[feature_cols],
            label=train_df[target_col],
            cat_features=CAT_FEATURES,
        )

        # FitAll: one trait-specific model trained across all environments.
        whole_model = make_model(args)
        whole_model.fit(whole_pool)

        # FT stage 1: keep a separate global model, as in the Rice script.
        first_model = make_model(args)
        first_model.fit(whole_pool)

        for round_number, (year, location) in enumerate(environments, start=1):
            print("-" * 80)
            print("Trait:", target_col, "| year:", year, "| location:", location)

            train_mask = (
                (train_df["year"] == year)
                & (train_df["location"] == location)
            )
            test_mask = (
                (test_df["year"] == year)
                & (test_df["location"] == location)
            )
            train_data = train_df.loc[train_mask]
            test_data = test_df.loc[test_mask]
            if train_data.empty or test_data.empty:
                raise ValueError(
                    f"Empty train/test environment for {target_col}: "
                    f"year={year}, location={location}"
                )

            train_pool = Pool(
                data=train_data[feature_cols],
                label=train_data[target_col],
                cat_features=CAT_FEATURES,
            )
            test_pool = Pool(
                data=test_data[feature_cols],
                label=test_data[target_col],
                cat_features=CAT_FEATURES,
            )
            y_true = test_data[target_col].to_numpy()

            whole_metrics = safe_metrics(y_true, whole_model.predict(test_pool))

            # FitOne: an independent model for this location-year environment.
            one_model = make_model(args)
            one_model.fit(train_pool)
            one_metrics = safe_metrics(y_true, one_model.predict(test_pool))

            # FT: append local trees to the separately trained global model.
            ft_model = make_model(args)
            ft_model.fit(train_pool, init_model=first_model)
            ft_metrics = safe_metrics(y_true, ft_model.predict(test_pool))

            results.append(
                {
                    "Trait": target_col,
                    "Round": round_number,
                    "year": year,
                    "location": location,
                    "test_data": len(test_data),
                    "TraingWhole_R2": whole_metrics[0],
                    "TraingWhole_RMSE": whole_metrics[1],
                    "TraingWhole_PCC": whole_metrics[2],
                    "TraingOnebyOne_R2": one_metrics[0],
                    "TraingOnebyOne_RMSE": one_metrics[1],
                    "TraingOnebyOne_PCC": one_metrics[2],
                    "TraingTwice_R2": ft_metrics[0],
                    "TraingTwice_RMSE": ft_metrics[1],
                    "TraingTwice_PCC": ft_metrics[2],
                }
            )

    return pd.DataFrame(results)


def validate_inputs(input_dir: Path) -> None:
    total_environments = 0
    for target_col in TRAITS:
        train_df, test_df, feature_cols, environments = read_trait_data(
            input_dir, target_col
        )
        total_environments += len(environments)
        print(
            f"Validated {target_col}: train={len(train_df)}, test={len(test_df)}, "
            f"features={len(feature_cols)}, environments={len(environments)}"
        )
    print(f"Validated {len(TRAITS)} traits and {total_environments} tasks.")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    if args.validate_only:
        validate_inputs(input_dir)
        return

    output = (
        args.output.expanduser().resolve()
        if args.output
        else input_dir / DEFAULT_OUTPUT_NAME
    )
    result_df = run_experiment(args, input_dir)
    print(result_df)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print("Final results saved to:", output)


if __name__ == "__main__":
    main()
