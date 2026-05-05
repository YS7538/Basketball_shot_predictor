from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = DATA_DIR / "data.csv"
OUTPUT_DIR = DATA_DIR / "processed"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "shot_made_flag"

NUMERIC_FEATURES = [
    "loc_x",
    "loc_y",
    "minutes_remaining",
    "seconds_remaining",
    "period",
    "playoffs",
    "shot_distance",
    "seconds_left_in_period",
    "game_year",
    "game_month",
    "game_dayofweek",
]

CATEGORICAL_FEATURES = [
    "action_type",
    "combined_shot_type",
    "season",
    "shot_type",
    "shot_zone_area",
    "shot_zone_basic",
    "shot_zone_range",
    "opponent",
    "home_away",
]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year
    df["game_month"] = df["game_date"].dt.month
    df["game_dayofweek"] = df["game_date"].dt.dayofweek

    df["seconds_left_in_period"] = (
        df["minutes_remaining"] * 60 + df["seconds_remaining"]
    )
    df["home_away"] = df["matchup"].str.contains("vs.", regex=False).map(
        {True: "home", False: "away"}
    )

    return df


def split_labeled_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = df[df[TARGET].notna()].copy()
    unlabeled = df[df[TARGET].isna()].copy()

    labeled[TARGET] = labeled[TARGET].astype(int)

    return labeled, unlabeled


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def transform_to_dataframe(
    preprocessor: ColumnTransformer, features: pd.DataFrame
) -> pd.DataFrame:
    transformed = preprocessor.transform(features)
    columns = preprocessor.get_feature_names_out()

    return pd.DataFrame(transformed, columns=columns, index=features.index)


def save_outputs(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    x_unlabeled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    unlabeled: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    x_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    x_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    x_unlabeled.to_csv(OUTPUT_DIR / "X_unlabeled.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False, header=[TARGET])
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False, header=[TARGET])

    unlabeled[["shot_id"]].to_csv(OUTPUT_DIR / "unlabeled_shot_ids.csv", index=False)
    unlabeled[["shot_id", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].to_csv(
        OUTPUT_DIR / "unlabeled_shots.csv", index=False
    )
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")


def main() -> None:
    df = add_features(load_data())
    labeled, unlabeled = split_labeled_unlabeled(df)

    features = labeled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = labeled[TARGET]

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(x_train_raw)

    x_train = transform_to_dataframe(preprocessor, x_train_raw)
    x_test = transform_to_dataframe(preprocessor, x_test_raw)
    x_unlabeled = transform_to_dataframe(
        preprocessor, unlabeled[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )

    save_outputs(
        preprocessor, x_train, x_test, x_unlabeled, y_train, y_test, unlabeled
    )

    print("Preprocessing complete.")
    print(f"Labeled rows used for training: {len(labeled)}")
    print(f"Unlabeled rows saved for later prediction: {len(unlabeled)}")
    print(f"Training features: {x_train.shape}")
    print(f"Testing features: {x_test.shape}")
    print(f"Unlabeled prediction features: {x_unlabeled.shape}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
