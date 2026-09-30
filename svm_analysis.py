from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = Path(__file__).with_name("wdbc.data")
RANDOM_STATE = 42
CV_FOLDS = 5
C_VALUES = [0.01, 0.1, 1, 10, 100]
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
}


def make_model(**svc_parameters):
    return make_pipeline(StandardScaler(), SVC(**svc_parameters))


def evaluate_metrics(model, features, target):
    predictions = model.predict(features)
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions),
        "recall": recall_score(target, predictions),
        "f1": f1_score(target, predictions),
    }


def cross_validation_summary(features, target, model):
    scores = cross_validate(
        model,
        features,
        target,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
    )
    return {
        metric: scores[f"test_{metric}"].mean()
        for metric in SCORING
    }


def main():
    dataset = pd.read_csv(DATA_PATH, header=None)
    features = dataset.iloc[:, 2:]
    target = dataset.iloc[:, 1].map({"B": 0, "M": 1})

    if target.isna().any():
        raise ValueError("Unexpected target label; expected only 'B' and 'M'.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    baseline = make_model()
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate_metrics(baseline, x_test, y_test)
    print("Baseline SVM (default SVC parameters; held-out test set)")
    print(pd.Series(baseline_metrics).to_string(float_format="{:.4f}".format))
    print("Confusion matrix (rows=true [B, M], columns=predicted [B, M]):")
    print(confusion_matrix(y_test, baseline.predict(x_test), labels=[0, 1]))

    c_results = []
    for c_value in C_VALUES:
        result = cross_validation_summary(
            x_train, y_train, make_model(C=c_value)
        )
        c_results.append({"C": c_value, **result})
    print("\nC comparison (5-fold CV on training set; kernel='rbf')")
    c_table = pd.DataFrame(c_results).set_index("C")
    print(c_table.to_string(float_format="{:.4f}".format))
    figure, axis = plt.subplots(figsize=(8, 5))
    for metric in SCORING:
        axis.plot(c_table.index, c_table[metric], marker="o", label=metric.title())
    axis.set_xscale("log")
    axis.set_xticks(C_VALUES, labels=[str(value) for value in C_VALUES])
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("C (regularization parameter, log scale)")
    axis.set_ylabel("Mean 5-fold validation score")
    axis.set_title("Effect of C on SVM Validation Performance")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    plot_path = Path(__file__).with_name("c_validation_performance.png")
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(f"\nC validation plot saved to: {plot_path}")
    best_c = c_table["f1"].idxmax()
    print(
        f"Highest mean CV F1 was at C={best_c:g}; "
        f"C=0.01 yielded F1={c_table.loc[0.01, 'f1']:.4f}."
    )
    manual_model = make_model(C=float(best_c))
    manual_model.fit(x_train, y_train)
    manual_metrics = evaluate_metrics(manual_model, x_test, y_test)

    kernel_results = []
    for kernel in ["linear", "poly", "rbf"]:
        result = cross_validation_summary(
            x_train, y_train, make_model(kernel=kernel)
        )
        kernel_results.append({"kernel": kernel, **result})
    print("\nKernel comparison (5-fold CV on training set; default C and gamma)")
    kernel_table = pd.DataFrame(kernel_results).set_index("kernel")
    print(kernel_table.to_string(float_format="{:.4f}".format))
    axis = kernel_table.plot(kind="bar", figsize=(8, 5), ylim=(0, 1.05))
    axis.set_xlabel("SVM kernel")
    axis.set_ylabel("Mean 5-fold validation score")
    axis.set_title("Validation Performance by SVM Kernel")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    axis.legend(title="Metric")
    axis.figure.tight_layout()
    kernel_plot_path = Path(__file__).with_name("kernel_validation_performance.png")
    axis.figure.savefig(kernel_plot_path, dpi=160)
    plt.close(axis.figure)
    print(f"Kernel comparison plot saved to: {kernel_plot_path}")
    best_kernel = kernel_table["f1"].idxmax()
    print(
        f"Highest mean CV F1 was with the {best_kernel} kernel; "
        f"poly precision was {kernel_table.loc['poly', 'precision']:.4f} but "
        f"recall was only {kernel_table.loc['poly', 'recall']:.4f}, indicating "
        "it missed more malignant cases."
    )

    search = GridSearchCV(
        make_model(),
        param_grid=[
            {"svc__kernel": ["linear"], "svc__C": C_VALUES},
            {
                "svc__kernel": ["poly", "rbf"],
                "svc__C": C_VALUES,
                "svc__gamma": [0.001, 0.01, 0.1, 1],
            },
        ],
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    tuned_metrics = evaluate_metrics(search.best_estimator_, x_test, y_test)
    print("\nGridSearchCV (5-fold CV; optimized for malignant-class F1)")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best mean CV F1: {search.best_score_:.4f}")
    print("Tuned SVM (held-out test set)")
    print(pd.Series(tuned_metrics).to_string(float_format="{:.4f}".format))
    print("Confusion matrix (rows=true [B, M], columns=predicted [B, M]):")
    print(confusion_matrix(y_test, search.predict(x_test), labels=[0, 1]))

    comparison = pd.DataFrame(
        {
            "Default SVM": baseline_metrics,
            f"Manual C={best_c:g}": manual_metrics,
            "GridSearchCV": tuned_metrics,
        }
    ).T
    print("\nUntouched test-set comparison")
    print(comparison.to_string(float_format="{:.4f}".format))
    print("Confusion matrices (rows=true [B, M], columns=predicted [B, M]):")
    for model_name, model in [
        ("Default SVM", baseline),
        (f"Manual C={best_c:g}", manual_model),
        ("GridSearchCV", search.best_estimator_),
    ]:
        print(f"{model_name}:\n{confusion_matrix(y_test, model.predict(x_test), labels=[0, 1])}")


if __name__ == "__main__":
    main()