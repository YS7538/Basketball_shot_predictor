from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "Dataset" / "processed"
OUTPUT_DIR = PROJECT_DIR / "models"

RANDOM_STATE = 42


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["shot_made_flag"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["shot_made_flag"]

    return x_train, x_test, y_train, y_test


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_compare_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, object, pd.Series]:
    results = []
    trained_models = {}
    test_predictions = {}

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "f1_score": f1,
            }
        )
        trained_models[model_name] = model
        test_predictions[model_name] = predictions

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"], ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = test_predictions[best_model_name]

    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    return best_model_name, best_model, best_predictions


def save_confusion_matrix(y_test: pd.Series, predictions: pd.Series) -> None:
    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Missed", "Actual Made"],
        columns=["Predicted Missed", "Predicted Made"],
    )
    matrix_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Missed", "Made"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Basketball Shot Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def save_unlabeled_predictions(model: object) -> None:
    x_unlabeled_path = PROCESSED_DIR / "X_unlabeled.csv"
    shot_ids_path = PROCESSED_DIR / "unlabeled_shot_ids.csv"

    if not x_unlabeled_path.exists() or not shot_ids_path.exists():
        return

    x_unlabeled = pd.read_csv(x_unlabeled_path)
    shot_ids = pd.read_csv(shot_ids_path)
    probabilities = model.predict_proba(x_unlabeled)[:, 1]
    predictions = model.predict(x_unlabeled)

    output = shot_ids.copy()
    output["predicted_shot_made_flag"] = predictions
    output["shot_made_probability"] = probabilities
    output.to_csv(OUTPUT_DIR / "unlabeled_predictions.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    x_train, x_test, y_train, y_test = load_processed_data()

    best_model_name, best_model, predictions = train_and_compare_models(
        x_train, x_test, y_train, y_test
    )
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nBest model: {best_model_name}")
    print(f"Best model accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["Missed", "Made"]))

    save_confusion_matrix(y_test, predictions)
    save_unlabeled_predictions(best_model)
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    print(f"\nSaved model and outputs to: {OUTPUT_DIR}")
    print("Model comparison saved as model_comparison.csv")
    print("Confusion matrix saved as confusion_matrix.csv and confusion_matrix.png")


if __name__ == "__main__":
    main()
