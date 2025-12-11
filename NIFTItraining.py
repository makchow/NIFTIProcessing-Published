"""
NIFTItrainingNestedCV.py
-----------------------

This script implements a robust, modular machine learning pipeline for tabular data (e.g., radiomics features) with support for nested cross-validation (nested CV) for unbiased model selection and evaluation.

Key Features:
- Modular pipeline classes for Random Forest and Logistic Regression, easily extensible to other models.
- Automatic feature selection (SelectKBest) and preprocessing (imputation, scaling, variance thresholding).
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Nested cross-validation: outer loop for unbiased test evaluation, inner loop for hyperparameter search.
- Centralized saving and reporting of all metrics, including nested CV results.

Usage:
- Place your patient feature CSV in the expected location (see FEATURES_CSV).
- Customize the models list in main() to add/remove models.
- Run the script. Results and metrics will be saved in the trainingML/ subdirectory.

Class/Function Documentation:
-----------------------------

BaseMLPipeline:
    - Base class for ML pipelines. Handles pipeline construction, feature selection, and search_fit (hyperparameter tuning).
    - Methods:
        - __init__: Sets up the pipeline steps and estimator.
        - search_fit: Runs GridSearchCV or RandomizedSearchCV for hyperparameter tuning. Stores best estimator and metrics.
        - save: Saves the trained pipeline and all metrics, including nested CV results if provided.

RandomForestMLPipeline, LogisticRegressionMLPipeline:
    - Subclasses of BaseMLPipeline with default estimators and parameter grids for each model type.

nested_cv_evaluate:
    - Runs nested cross-validation for a given pipeline and dataset.
    - Outer loop: splits data into train/test folds for unbiased evaluation.
    - Inner loop: runs hyperparameter search (search_fit) on the training fold.
    - Returns test accuracies and best parameters for each outer fold.

main:
    - Loads data and defines models.
    - Runs nested CV for each model, prints and saves results using the pipeline's save method.

Pipeline Steps:
    - Imputation (median), scaling (StandardScaler), variance thresholding, feature selection (SelectKBest), classifier (clf).
    - 'clf' is the final classifier step (e.g., RandomForestClassifier or LogisticRegression).

Outputs:
    - For each model, saves a metrics file with:
        - Best parameters (if available)
        - Accuracy, confusion matrix, classification report, top features (if available)
        - Nested CV results: mean and per-fold test accuracy, best params per fold

Best Practices:
    - Nested CV provides an unbiased estimate of model performance, especially important for small datasets.
    - All reporting and saving is centralized for reproducibility and clarity.

"""

import os
import numpy as np
import pandas as pd
import warnings
import NIFTIpatient
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import random
from collections import Counter, defaultdict

import xgboost as xgb  # For future use
from xgboost import XGBClassifier, XGBRFClassifier

# For ROC curve plotting
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

# Filter out expected warnings from scikit-learn
# This occurs during feature selection when features have zero variance or cause numerical issues
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)
# This was about the data format, which we fixed but the warning filters provide an extra safety net
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*column-vector y was passed.*"
)
# This occurs when a class has no predicted samples, which is common in small datasets
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*Precision is ill-defined.*"
)


# Standalone class for plotting ROC curve for binary classifiers
class ROCCurvePlotter:
    """
    Utility class to plot ROC curve for binary classifiers.
    Usage:
        plotter = ROCCurvePlotter(y_true, y_score, title="ROC Curve")
        plotter.plot()
    Args:
        y_true: array-like of shape (n_samples,) - True binary labels (0 or 1)
        y_score: array-like of shape (n_samples,) - Target scores, can be probability estimates of the positive class
        title: str - Title for the plot
    """

    def __init__(self, y_true, y_score, title="ROC Curve"):
        """
        Initialize with true labels and predicted scores.

        Args:
            y_true (array-like of shape (n_samples,): True binary labels (0 or 1)
            y_score (array-like of shape (n_samples,): Target scores, can be probability estimates of the positive class
            title (str): Title for the plot
        Returns:
            None
        """
        self.y_true = y_true
        self.y_score = y_score
        self.title = title
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.roc_auc = None

    def compute(self):
        """
        Computes FPR, TPR, thresholds, and AUC.

        Args:
            None

        Returns:
            None
        """
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_score)
        self.roc_auc = auc(self.fpr, self.tpr)

    def plot(self, show=True, save_path=None):
        """
        Plots the ROC curve.

        Args:
            show (bool): If True, display the plot interactively.
            save_path (str or None): If provided, save the plot to this path.

        Returns:
            None
        """
        self.compute()
        plt.figure()
        plt.plot(
            self.fpr,
            self.tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {self.roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(self.title)
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_nested_cv_roc_curves(
        roc_data, save_path=None, title="Nested CV ROC Curves", csv_path=None
    ):
        """
        Plot all ROC curves from nested CV runs on the same graph.

        Args:
            roc_data (list of tuples): list of (fpr, tpr, auc, fold) tuples
            save_path (str or None): if provided, save the plot to this path
            title (str): plot title
            csv_path (str or None): if provided, save all ROC data to this CSV for reproducibility

        Returns:
            None
        """

        plt.figure(figsize=(8, 6))
        aucs = []
        # Optionally save all ROC data to CSV for reproducibility
        if csv_path is not None:
            # Flatten all ROC data into a DataFrame
            rows = []
            for fpr, tpr, roc_auc, fold in roc_data:
                for f, t in zip(fpr, tpr):
                    rows.append({"fold": fold, "fpr": f, "tpr": t, "auc": roc_auc})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        for fpr, tpr, roc_auc, fold in roc_data:
            plt.plot(fpr, tpr, lw=2, label=f"Fold {fold} (AUC = {roc_auc:.2f})")
            aucs.append(roc_auc)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1)
        n = len(aucs)
        # 95% CI for mean: mean ± 1.96 * (std / sqrt(n))
        if n > 1:
            ci95 = 1.96 * (std_auc / np.sqrt(n))
            ci_low = mean_auc - ci95
            ci_high = mean_auc + ci95
            ci_str = f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]"
        else:
            ci_str = ""
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title_str = f"{title}\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}"
        if ci_str:
            title_str += f"\n{ci_str}"
        plt.title(title_str)
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        # Do not show the plot interactively
        plt.close()

    @staticmethod
    def plot_holdout_roc_curves_across_seeds(
        roc_data, save_path=None, title="Holdout ROC Curves Across Seeds", csv_path=None
    ):
        """
        Plot all holdout ROC curves from different seeds for a single model.

        Args:
            roc_data (list of tuples): list of (fpr, tpr, auc, seed) tuples
            save_path (str or None): if provided, save the plot to this path
            title (str): plot title
            csv_path (str or None): if provided, save all ROC data to this CSV for reproducibility

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        aucs = []
        # Optionally save all ROC data to CSV for reproducibility
        if csv_path is not None:
            rows = []
            for fpr, tpr, roc_auc, seed in roc_data:
                for f, t in zip(fpr, tpr):
                    rows.append({"seed": seed, "fpr": f, "tpr": t, "auc": roc_auc})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
        for fpr, tpr, roc_auc, seed in roc_data:
            plt.plot(fpr, tpr, lw=2, label=f"Seed {seed} (AUC = {roc_auc:.2f})")
            aucs.append(roc_auc)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1)
        n = len(aucs)
        # 95% CI for mean: mean ± 1.96 * (std / sqrt(n))
        if n > 1:
            ci95 = 1.96 * (std_auc / np.sqrt(n))
            ci_low = mean_auc - ci95
            ci_high = mean_auc + ci95
            ci_str = f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]"
        else:
            ci_str = ""
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        title_str = f"{title}\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}"
        if ci_str:
            title_str += f"\n{ci_str}"
        plt.title(title_str)
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)
        plt.close()


class BaseMLPipeline:
    def get_preprocessed_feature_names(self, pipeline, X):
        """
        Retrieve feature names after preprocessing, with fallback and prefix stripping.

        Args:
            pipeline (skLearn Pipeline object): The fitted pipeline.
            X (pandas DataFrame): Original input features.
        Returns:
            feature_names (list of str): List of feature names after preprocessing.
        """

        if "preprocess" in pipeline.named_steps:
            try:
                X_preprocessed = pipeline.named_steps["preprocess"].transform(X)
                if hasattr(pipeline.named_steps["preprocess"], "get_feature_names_out"):
                    feature_names = pipeline.named_steps[
                        "preprocess"
                    ].get_feature_names_out()
                else:
                    feature_names = [f"f{i}" for i in range(X_preprocessed.shape[1])]
            except Exception:
                feature_names = X.columns
        else:
            feature_names = X.columns
        # Strip 'remainder__' prefix
        feature_names = [f.replace("remainder__", "") for f in feature_names]
        return feature_names

    """
    BaseMLPipeline: A flexible base class for building and running ML pipelines on tabular data.
    Customize the pipeline steps, estimator, and feature selection as needed for your use case.
    """

    def __init__(
        self,
        n_features=None,
        estimator=None,
        random_state=None,
        ord_categories=None,
    ):
        """
        Initialize the ML pipeline with preprocessing, feature selection, and estimator.

        Args:
            n_features (int or None): Number of top features to select (SelectKBest). If None, 'all' is used.
            estimator (sklearn estimator): The final estimator (classifier/regressor) to use.
            random_state (int or None): Random state for reproducibility.
            ord_categories (dict or None): Categorical feature categories for ordinal encoding (if needed).

        Returns:
            None
        """

        # If n_features is None, we'll let GridSearch tune select__k and default SelectKBest to 'all'
        self.n_features = n_features
        self.random_state = random_state
        # Define categorical columns for mhhcohort data (excluding target variable 'pts')
        self.ord_categories = ord_categories

        # Treat all features as numeric (already encoded). Single median imputation.
        # IF CATEGORICAL FEATURES ARE PRESENT, modify this step to include OrdinalEncoder
        preprocessor = SimpleImputer(strategy="median")

        pipeline_steps = [
            ("preprocess", preprocessor),
            ("scale", StandardScaler()),
            ("var_thresh", VarianceThreshold(threshold=0.01)),
            (
                "select",
                SelectKBest(
                    score_func=f_classif,
                    k=(self.n_features if self.n_features is not None else "all"),
                ),
            ),
        ]

        self.estimator = estimator
        pipeline_steps.append(("clf", estimator))
        self.pipeline = Pipeline(pipeline_steps)
        self.selected_features_ = None
        self.importances_ = None
        self.metrics_ = None

    def search_fit(
        self,
        X,
        y,
        random_state,
        search_type="grid",
        param_grid=None,
        n_iter=10,
        cv=5,
        scoring="accuracy",
    ):
        """
        Perform hyperparameter search (GridSearchCV or RandomizedSearchCV) and fit the pipeline.

        Args:
            X (array-like of shape (n_samples, n_features)): Input features.
            y (array-like of shape (n_samples,)): Target labels.
            random_state (int): Random state for reproducibility.
            search_type (str): "grid" for GridSearchCV, "random" for RandomizedSearchCV.
            param_grid (dict): Parameter grid or distributions for search.
            n_iter (int): Number of iterations for RandomizedSearchCV (ignored for GridSearchCV).
            cv (int): Number of cross-validation folds.
            scoring (str): Scoring metric for evaluation.

        Returns:
            search (sklearn GridSearchCV or RandomizedSearchCV object): Fitted search object.
        """

        if search_type == "grid":
            search = GridSearchCV(
                self.pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
        elif search_type == "random":
            search = RandomizedSearchCV(
                self.pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError("search_type must be 'grid' or 'random'")

        search.fit(X, y)
        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X)

        cm = confusion_matrix(y, y_pred)
        # Specificity: TN / (TN + FP)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        else:
            specificity = np.nan
        # AUC: only if y has both classes and proba/decision available
        try:
            if hasattr(best_pipeline.named_steps["clf"], "predict_proba"):
                y_score = best_pipeline.named_steps["clf"].predict_proba(X)[:, 1]
            else:
                y_score = best_pipeline.named_steps["clf"].decision_function(X)
            auc_val = roc_auc_score(y, y_score)
        except Exception:
            auc_val = np.nan
        metrics = {
            "best_params": search.best_params_,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc": auc_val,
            "confusion_matrix": cm,
            "classification_report": classification_report(y, y_pred),
        }
        if (
            hasattr(best_pipeline.named_steps["clf"], "feature_importances_")
            and "select" in best_pipeline.named_steps
        ):
            # Get feature names after preprocessing
            preprocessed_feature_names = self.get_preprocessed_feature_names(
                best_pipeline, X
            )

            # Get the feature names after variance threshold but before selection
            if "var_thresh" in best_pipeline.named_steps:
                var_thresh_mask = best_pipeline.named_steps["var_thresh"].get_support()
                features_after_var_thresh = np.array(preprocessed_feature_names)[
                    var_thresh_mask
                ]
            else:
                features_after_var_thresh = preprocessed_feature_names

            # Get the selection mask and apply it to the reduced feature set
            selected_mask = best_pipeline.named_steps["select"].get_support()
            selected_features = features_after_var_thresh[selected_mask]
            importances = best_pipeline.named_steps["clf"].feature_importances_
            importances_series = pd.Series(importances, index=selected_features)
            metrics["top_features"] = importances_series.sort_values(ascending=False)
        elif (
            hasattr(best_pipeline.named_steps["clf"], "coef_")
            and "select" in best_pipeline.named_steps
        ):
            # Get feature names after preprocessing
            preprocessed_feature_names = self.get_preprocessed_feature_names(
                best_pipeline, X
            )

            # Get the feature names after variance threshold but before selection
            if "var_thresh" in best_pipeline.named_steps:
                var_thresh_mask = best_pipeline.named_steps["var_thresh"].get_support()
                features_after_var_thresh = np.array(preprocessed_feature_names)[
                    var_thresh_mask
                ]
            else:
                features_after_var_thresh = preprocessed_feature_names

            # Get the selection mask and apply it to the reduced feature set
            selected_mask = best_pipeline.named_steps["select"].get_support()
            selected_features = features_after_var_thresh[selected_mask]
            # Use absolute values of coefficients for importance
            coefs = np.abs(best_pipeline.named_steps["clf"].coef_[0])
            importances_series = pd.Series(coefs, index=selected_features)
            metrics["top_features"] = importances_series.sort_values(ascending=False)
        self.metrics_ = metrics
        self.pipeline = best_pipeline
        return search

    def save(self, model_path=None, metrics_path=None, nested_cv_results=None):
        """
        Save the trained pipeline and comprehensive metrics report.

        Args:
            model_path (str or None): If provided, save the trained pipeline to this path.
            metrics_path (str or None): If provided, save the comprehensive metrics report to this path.
            nested_cv_results (tuple or None): If provided, should be (test_scores, best_params_list, model_name)

        Returns:
            None
        """
        if model_path is not None:
            joblib.dump(self.pipeline, model_path)
        if metrics_path is not None:
            with open(metrics_path, "w") as f:
                f.write("ML Pipeline Comprehensive Report\n")
                f.write("=" * 50 + "\n\n")

                # Nested CV Results (Unbiased Performance Estimate)
                if nested_cv_results is not None:
                    test_scores, best_params_list, model_name = nested_cv_results
                    f.write("NESTED CROSS-VALIDATION RESULTS (Unbiased Performance)\n")
                    f.write("-" * 55 + "\n")
                    f.write(
                        f"Mean test accuracy: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}\n"
                    )
                    f.write(f"All test accuracies: {test_scores}\n")
                    f.write(f"Best hyperparameters per fold:\n")
                    for i, params in enumerate(best_params_list):
                        f.write(f"  Fold {i+1}: {params}\n")
                    f.write("\n")

                # Final / Hold-out Results (Feature Interpretation)
                if self.metrics_ is not None:
                    f.write("FINAL MODEL RESULTS\n")
                    f.write("-" * 45 + "\n")

                    if "best_params" in self.metrics_:
                        f.write(
                            f"Best hyperparameters: {self.metrics_['best_params']}\n"
                        )

                    # Training metrics (if provided)
                    if any(k.startswith("train_") for k in self.metrics_.keys()):
                        f.write("\nTRAINING SET PERFORMANCE\n")
                        f.write("~" * 30 + "\n")
                        if self.metrics_.get("train_accuracy") is not None:
                            f.write(
                                f"Accuracy: {self.metrics_['train_accuracy']:.3f}\n"
                            )
                        if self.metrics_.get("train_confusion_matrix") is not None:
                            f.write("Confusion Matrix:\n")
                            f.write(str(self.metrics_["train_confusion_matrix"]) + "\n")
                        if self.metrics_.get("train_classification_report") is not None:
                            f.write("Classification Report:\n")
                            f.write(self.metrics_["train_classification_report"] + "\n")
                        # Add extra metrics if available
                        for k in [
                            "train_precision",
                            "train_recall",
                            "train_specificity",
                            "train_f1",
                            "train_auc",
                        ]:
                            if self.metrics_.get(k) is not None:
                                f.write(
                                    f"{k.replace('train_','').capitalize()}: {self.metrics_[k]:.3f}\n"
                                )
                    else:
                        # Backward compatibility: older single-set metrics
                        if self.metrics_.get("accuracy") is not None:
                            f.write(
                                f"Training accuracy: {self.metrics_.get('accuracy'):.3f}\n"
                            )
                        if self.metrics_.get("precision") is not None:
                            f.write(
                                f"Precision: {self.metrics_.get('precision'):.3f}\n"
                            )
                        if self.metrics_.get("recall") is not None:
                            f.write(f"Recall: {self.metrics_.get('recall'):.3f}\n")
                        if self.metrics_.get("specificity") is not None:
                            f.write(
                                f"Specificity: {self.metrics_.get('specificity'):.3f}\n"
                            )
                        if self.metrics_.get("f1") is not None:
                            f.write(f"F1 score: {self.metrics_.get('f1'):.3f}\n")
                        if self.metrics_.get("auc") is not None:
                            f.write(f"AUC: {self.metrics_.get('auc'):.3f}\n")
                        if self.metrics_.get("confusion_matrix") is not None:
                            f.write("Confusion Matrix (Training Data):\n")
                            f.write(str(self.metrics_.get("confusion_matrix")) + "\n")
                        if self.metrics_.get("classification_report") is not None:
                            f.write("Classification Report (Training Data):\n")
                            f.write(self.metrics_.get("classification_report") + "\n")

                    # Test metrics (hold-out set)
                    if any(k.startswith("test_") for k in self.metrics_.keys()):
                        f.write("\nHOLD-OUT TEST SET PERFORMANCE (20%)\n")
                        f.write("~" * 45 + "\n")
                        if self.metrics_.get("test_accuracy") is not None:
                            f.write(f"Accuracy: {self.metrics_['test_accuracy']:.3f}\n")
                        if self.metrics_.get("test_precision") is not None:
                            f.write(
                                f"Precision: {self.metrics_['test_precision']:.3f}\n"
                            )
                        if self.metrics_.get("test_recall") is not None:
                            f.write(f"Recall: {self.metrics_['test_recall']:.3f}\n")
                        if self.metrics_.get("test_specificity") is not None:
                            f.write(
                                f"Specificity: {self.metrics_['test_specificity']:.3f}\n"
                            )
                        if self.metrics_.get("test_f1") is not None:
                            f.write(f"F1 score: {self.metrics_['test_f1']:.3f}\n")
                        if self.metrics_.get("test_auc") is not None:
                            f.write(f"ROC AUC: {self.metrics_['test_auc']:.3f}\n")
                        if self.metrics_.get("test_confusion_matrix") is not None:
                            f.write("Confusion Matrix:\n")
                            f.write(str(self.metrics_["test_confusion_matrix"]) + "\n")
                        if self.metrics_.get("test_classification_report") is not None:
                            f.write("Classification Report:\n")
                            f.write(self.metrics_["test_classification_report"] + "\n")

                    # Feature importance / coefficients
                    if self.metrics_.get("top_features", None) is not None:
                        all_feats_series = self.metrics_["top_features"]
                        # Preview
                        f.write("\nTOP FEATURES PREVIEW (first 20 shown):\n")
                        f.write("-" * 45 + "\n")
                        preview = all_feats_series.head(20)
                        f.write(preview.to_string() + "\n\n")
                        # Full list
                        f.write("FULL RANKED FEATURE LIST:\n")
                        f.write("-" * 45 + "\n")
                        f.write(all_feats_series.to_string() + "\n\n")
                        selected_features = all_feats_series.index.tolist()
                        f.write("ALL SELECTED FEATURES (ordered):\n")
                        f.write("-" * 45 + "\n")
                        for i, feature in enumerate(selected_features, 1):
                            f.write(f"{i:3d}. {feature}\n")

        if model_path is not None:
            print(f"Final model saved to: {model_path}")
        if metrics_path is not None:
            print(f"Comprehensive metrics saved to: {metrics_path}")


class RandomForestMLPipeline(BaseMLPipeline):
    def __init__(
        self, n_features=None, random_state=None, ord_categories=None, param_grid=None
    ):
        """
        Initialize RandomForestMLPipeline with default RandomForestClassifier and parameter grid.

        Args:
            n_features (int or None): Number of top features to select (SelectKBest). If None, 'all' is used.
            random_state (int or None): Random state for reproducibility.
            ord_categories (dict or None): Categorical feature categories for ordinal encoding (if needed).
            param_grid (dict or None): Parameter grid for hyperparameter search. If None, default grid is used.

        Returns:
            None
        """
        super().__init__(
            n_features=n_features,
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1
            ),
            random_state=random_state,
            ord_categories=ord_categories,
        )
        # Default or user-supplied param grid for search
        self.param_grid = param_grid or {
            "select__k": [10, 20, 30, 40, 50, 60, 70, 80, 100],
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
        }


class LogisticRegressionMLPipeline(BaseMLPipeline):
    def __init__(
        self, n_features=None, random_state=None, ord_categories=None, param_grid=None
    ):
        """
        Initialize LogisticRegressionMLPipeline with default LogisticRegression and parameter grid.

        Args:
            n_features (int or None): Number of top features to select (SelectKBest). If None, 'all' is used.
            random_state (int or None): Random state for reproducibility.
            ord_categories (dict or None): Categorical feature categories for ordinal encoding (if needed).
            param_grid (dict or None): Parameter grid for hyperparameter search. If None, default grid is used.

        Returns:
            None
        """
        super().__init__(
            n_features=n_features,
            estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            random_state=random_state,
            ord_categories=ord_categories,
        )
        # Default or user-supplied param grid for search
        self.param_grid = param_grid or {
            "select__k": [10, 20, 30, 40],
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l2"],
        }


class XGBoostMLPipeline(BaseMLPipeline):
    def __init__(
        self, n_features=None, random_state=None, ord_categories=None, param_grid=None
    ):
        """
        Initialize XGBoostMLPipeline with default XGBClassifier and parameter grid.

        Args:
            n_features (int or None): Number of top features to select (SelectKBest). If None, 'all' is used.
            random_state (int or None): Random state for reproducibility.
            ord_categories (dict or None): Categorical feature categories for ordinal encoding (if needed).
            param_grid (dict or None): Parameter grid for hyperparameter search. If None, default grid is used.

        Returns:
            None

        """

        super().__init__(
            n_features=n_features,
            estimator=XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            random_state=random_state,
            ord_categories=ord_categories,
        )
        # Default or user-supplied param grid for search
        self.param_grid = param_grid or {
            "select__k": [10, 20, 30, 40, 50],
            "clf__max_depth": [3, 5, 7],
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        }


class XGBoostRFMLPipeline(BaseMLPipeline):
    def __init__(
        self, n_features=None, random_state=None, ord_categories=None, param_grid=None
    ):
        """
        Initialize XGBoostRFMLPipeline with default XGBRFClassifier and parameter grid.

        Args:
            n_features (int or None): Number of top features to select (SelectKBest). If None, 'all' is used.
            random_state (int or None): Random state for reproducibility.
            ord_categories (dict or None): Categorical feature categories for ordinal encoding (if needed).
            param_grid (dict or None): Parameter grid for hyperparameter search. If None, default grid is used.

        Returns:
            None
        """

        super().__init__(
            n_features=n_features,
            estimator=XGBRFClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            random_state=random_state,
            ord_categories=ord_categories,
        )
        # Default or user-supplied param grid for search
        self.param_grid = param_grid or {
            "select__k": [10, 20, 30, 40, 50],
            "clf__max_depth": [3, 5, 7],
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__subsample": [0.8, 1.0],
        }


def nested_cv_evaluate(pipeline, X, y, outer_cv=3, inner_cv=3, search_type="random"):
    """
    Perform nested cross-validation for unbiased model selection and evaluation.
    Returns a list of test scores and best params for each outer fold.

    Args:
        pipeline (BaseMLPipeline instance): The ML pipeline to evaluate.
        X (pandas DataFrame): Input features.
        y (array-like): Target labels.
        outer_cv (int): Number of outer CV folds.
        inner_cv (int): Number of inner CV folds.
        search_type (str): "grid" or "random" for hyperparameter search.

    Returns:
        test_scores (list of floats): Test accuracies for each outer fold.
        best_params_list (list of dicts): Best hyperparameters for each outer fold.
        roc_data (list of tuples [fpr,tpr,auc,fold]): ROC curve data for each outer fold.
    """

    outer = StratifiedKFold(
        random_state=pipeline.random_state, n_splits=outer_cv, shuffle=True
    )
    test_scores = []
    best_params_list = []
    roc_data = []  # List of (fpr, tpr, auc, fold) for each fold
    fold = 1
    for train_idx, test_idx in outer.split(X, y):
        print(f"Indices are {train_idx}, {test_idx}")
        print(f"\n=== Outer Fold {fold} ===")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Recreate a fresh model of the same type; do not pass n_features so select__k is tuned via param_grid
        model = type(pipeline)(
            random_state=pipeline.random_state, ord_categories=pipeline.ord_categories
        )
        param_grid = getattr(model, "param_grid", None)
        model.search_fit(
            X_train,
            y_train,
            random_state=pipeline.random_state,
            search_type=search_type,
            param_grid=param_grid,
            cv=inner_cv,
        )
        y_pred = model.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy (outer fold): {acc:.3f}")
        print(f"Best params (inner CV): {model.metrics_['best_params']}")
        print(f"Classification report:\n{classification_report(y_test, y_pred)}")
        test_scores.append(acc)
        best_params_list.append(model.metrics_["best_params"])

        # ROC curve for this fold
        if hasattr(model.pipeline.named_steps["clf"], "predict_proba"):
            y_score = model.pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = model.pipeline.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        roc_data.append((fpr, tpr, roc_auc, fold))
        fold += 1
    return test_scores, best_params_list, roc_data


def aggregate_seed_metrics(aggregate_records, ml_dir):
    """
    Aggregate per-seed performance across models and write raw + summary outputs.

    Artifacts written (if records exist):
      - aggregate_seed_metrics_raw.csv
      - aggregate_seed_metrics_summary.csv
      - aggregate_seed_metrics_summary.txt

    Uses sample standard deviation (ddof=1) for across-seed variability.

    Args:
        aggregate_records (list of dict):
            Each dict has keys:
                {'model': str, 'seed': int,
                 'nested_mean_acc': float,
                 'holdout_accuracy': float,
                 'holdout_auc': float}
        ml_dir (str): Base training output directory.

    Returns:
        summary_df (pandas.DataFrame or None): Summary dataframe or None if no records.
    """
    if not aggregate_records:
        print("No aggregate records to summarize.")
        return None

    agg_df = pd.DataFrame(aggregate_records)
    agg_csv = os.path.join(ml_dir, "aggregate_seed_metrics_raw.csv")
    agg_df.to_csv(agg_csv, index=False)

    # Compute summary stats for all metrics if present
    summary_rows = []
    for model_name, grp in agg_df.groupby("model"):
        n = len(grp)

        def ci95(mean, std, n):
            if n > 1:
                ci = 1.96 * (std / np.sqrt(n))
                return ci
            else:
                return np.nan

        # Helper to get mean, std, ci for a column if present
        def metric_stats(col):
            if col in grp:
                mean = grp[col].mean()
                std = grp[col].std(ddof=1)
                ci = ci95(mean, std, n)
                return mean, ci
            else:
                return np.nan, np.nan

        nested_mean_acc_mean, nested_mean_acc_ci = metric_stats("nested_mean_acc")
        holdout_acc_mean, holdout_acc_ci = metric_stats("holdout_accuracy")
        holdout_auc_mean, holdout_auc_ci = metric_stats("holdout_auc")
        holdout_precision_mean, holdout_precision_ci = metric_stats("holdout_precision")
        holdout_recall_mean, holdout_recall_ci = metric_stats("holdout_recall")
        holdout_specificity_mean, holdout_specificity_ci = metric_stats(
            "holdout_specificity"
        )
        holdout_f1_mean, holdout_f1_ci = metric_stats("holdout_f1")

        summary_rows.append(
            {
                "model": model_name,
                "seeds": grp.seed.tolist(),
                "n_seeds": n,
                "nested_mean_acc_mean": nested_mean_acc_mean,
                "nested_mean_acc_ci": nested_mean_acc_ci,
                "holdout_acc_mean": holdout_acc_mean,
                "holdout_acc_ci": holdout_acc_ci,
                "holdout_auc_mean": holdout_auc_mean,
                "holdout_auc_ci": holdout_auc_ci,
                "holdout_precision_mean": holdout_precision_mean,
                "holdout_precision_ci": holdout_precision_ci,
                "holdout_recall_mean": holdout_recall_mean,
                "holdout_recall_ci": holdout_recall_ci,
                "holdout_specificity_mean": holdout_specificity_mean,
                "holdout_specificity_ci": holdout_specificity_ci,
                "holdout_f1_mean": holdout_f1_mean,
                "holdout_f1_ci": holdout_f1_ci,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(ml_dir, "aggregate_seed_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_txt = os.path.join(ml_dir, "aggregate_seed_metrics_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("Aggregated Seed Performance Summary\n")
        f.write("=" * 40 + "\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"Model: {row['model']}\n")
            f.write(f"Seeds: {row['seeds']}\n")
            # Nested CV accuracy
            if not np.isnan(row["nested_mean_acc_ci"]):
                f.write(
                    f"Nested Mean Acc (mean, 95% CI): {row['nested_mean_acc_mean']:.3f} ± {row['nested_mean_acc_ci']:.3f} [{row['nested_mean_acc_mean']-row['nested_mean_acc_ci']:.3f}, {row['nested_mean_acc_mean']+row['nested_mean_acc_ci']:.3f}]\n"
                )
            else:
                f.write(f"Nested Mean Acc (mean): {row['nested_mean_acc_mean']:.3f}\n")

            # Holdout metrics
            def write_metric(label, mean, ci):
                if not np.isnan(ci):
                    f.write(
                        f"{label} (mean, 95% CI): {mean:.3f} ± {ci:.3f} [{mean-ci:.3f}, {mean+ci:.3f}]\n"
                    )
                else:
                    f.write(f"{label} (mean): {mean:.3f}\n")

            write_metric(
                "Hold-out Accuracy", row["holdout_acc_mean"], row["holdout_acc_ci"]
            )
            write_metric(
                "Hold-out Precision",
                row["holdout_precision_mean"],
                row["holdout_precision_ci"],
            )
            write_metric(
                "Hold-out Recall", row["holdout_recall_mean"], row["holdout_recall_ci"]
            )
            write_metric(
                "Hold-out Specificity",
                row["holdout_specificity_mean"],
                row["holdout_specificity_ci"],
            )
            write_metric(
                "Hold-out F1 Score", row["holdout_f1_mean"], row["holdout_f1_ci"]
            )
            write_metric("Hold-out AUC", row["holdout_auc_mean"], row["holdout_auc_ci"])
            f.write("-" * 40 + "\n")

    print(f"Aggregation raw CSV written to: {agg_csv}")
    print(f"Aggregation summary CSV written to: {summary_csv}")
    print(f"Aggregation summary TXT written to: {summary_txt}")
    return summary_df


def aggregate_feature_usage(feature_usage, ml_dir, top_n=None):
    """
    Aggregate feature usage across seeds for each model.

    Args:
        feature_usage (dict):
            Keys are model names (str).
            Values are lists of dicts with keys:
                {'seed': int, 'features': list of str (ordered by importance)}
        ml_dir (str): Base training output directory.
        top_n (int or None): If provided, only consider top N features from each seed

    Returns
        bool or None: True if aggregation done, None if no records.
    """

    if not feature_usage:
        print("No feature usage records to aggregate.")
        return None

    report_lines = ["Aggregated Feature Usage Across Models", "=" * 50, ""]

    for model_name, records in feature_usage.items():
        if not records:
            continue
        model_dir = os.path.join(ml_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Collect all features and ranks
        rank_maps = []  # list of dict(feature -> rank index starting at 1)
        for rec in records:
            feats = rec["features"]
            if top_n is not None:
                feats = feats[:top_n]
            rank_map = {f: i + 1 for i, f in enumerate(feats)}
            rank_maps.append(rank_map)

        # Count occurrences
        all_features = [f for rm in rank_maps for f in rm.keys()]
        counter = Counter(all_features)
        n_seeds = len(rank_maps)

        rows = []
        for feat, cnt in counter.items():
            ranks = [rm[feat] for rm in rank_maps if feat in rm]
            rows.append(
                {
                    "feature": feat,
                    "count": cnt,
                    "proportion": cnt / n_seeds,
                    "in_all_seeds": cnt == n_seeds,
                    "mean_rank": float(np.mean(ranks)),
                    "median_rank": float(np.median(ranks)),
                    "best_rank": int(np.min(ranks)),
                    "worst_rank": int(np.max(ranks)),
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["in_all_seeds", "count", "mean_rank"], ascending=[False, False, True]
        )
        out_csv = os.path.join(model_dir, "feature_usage_summary.csv")
        df.to_csv(out_csv, index=False)
        print(f"Feature usage summary written for model '{model_name}' -> {out_csv}")

        # Append to report
        report_lines.append(f"Model: {model_name}")
        report_lines.append(f"Seeds analyzed: {n_seeds}")
        report_lines.append("Top recurring features (first 15 shown):")
        head_df = df.head(15)
        report_lines.append(head_df.to_string(index=False))
        report_lines.append("-" * 50)

    # Write consolidated report
    consolidated_path = os.path.join(ml_dir, "feature_usage_across_models.txt")
    with open(consolidated_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Consolidated feature usage report written to: {consolidated_path}")
    return True


def main():
    """
    Main function to run and save multiple ML pipelines on patient features.

    Args:
        None

    Returns:
        None
    """

    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    FEATURES_CSV = os.path.join(BASE_FOLDER, "trainingML", "patient_features.csv")
    ML_DIR = os.path.join(BASE_FOLDER, "trainingML")
    os.makedirs(ML_DIR, exist_ok=True)

    exclude_col = ["subj_id", "mrn"]
    ground_truth = ["pts"]

    # Parameter Setup
    # Option A: All social determinant columns (and sex) are assumed already numerically encoded.
    # We therefore disable special categorical processing and treat them as numeric features.
    ORD_CATEGORIES = None

    # ------- CHANGE THIS FOR RUNS -------
    EXCLUDE_COL_MATCH = []
    # Can exclude columns that match any of these substrings
    # e.g. EXCLUDE_COL_MATCH = ["RL_diff","RL_ratio", "RL_absdiff"]
    # N_FEATURES removed: k will be tuned via param_grid select__k

    # def generate_seeds(num_seeds=10, seed_min=1, seed_max=10000):
    #     """Generate a list of random integers for use as SEEDS."""
    #     return random.sample(range(seed_min, seed_max + 1), num_seeds)
    # SEEDS = generate_seeds(num_seeds=25, seed_min=1, seed_max=10000)
    SEEDS = [
        # Seeds generated using Google RNG 1000-9999
        1292,
        4337,
        5590,
        3348,
        3930,
        2253,
        2400,
        7705,
        6993,
        7322,
    ]

    # param_grids for models
    random_forest_params = {
        "select__k": [10, 20, 30, 40, None],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 5, 10],
    }

    logistic_regression_params = {
        "select__k": [10, 20, 30, 40, None],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
    }

    xgboost_params = {
        "select__k": [10, 20, 30, 40, 50],
        "clf__max_depth": [3, 5, 7],
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    }

    xgboost_rf_params = {
        "select__k": [10, 20, 30, 40, 50],
        "clf__max_depth": [3, 5, 7],
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__subsample": [0.8, 1.0],
    }

    OUTER_CV = 5
    INNER_CV = 5

    # ------------------------------------

    X, y = NIFTIpatient.load_all_patients(
        FEATURES_CSV,
        exclude_col=exclude_col,
        ground_truth=ground_truth,
        exclude_col_match=EXCLUDE_COL_MATCH,
    )

    # Convert y to 1D array to avoid warnings
    y = y.values.ravel()

    # Replace sentinel -1 with NaN so median imputation can appropriately handle missingness.
    # (Assumes -1 represents missing / not on file; adjust if you want to keep it as a separate value.)
    X = X.replace(-1, np.nan)

    # Aggregation container: list of dict records
    aggregate_records = []
    # Feature usage across seeds per model: model -> list of {seed, features}
    feature_usage = defaultdict(list)
    # Holdout ROC data for summary plot: model -> list of (fpr, tpr, auc, seed)
    holdout_roc_data = defaultdict(list)

    for seed in SEEDS:
        # Define as many models as you want here
        models = [
            (
                "rf",
                RandomForestMLPipeline(
                    param_grid=random_forest_params,
                    random_state=seed,
                    ord_categories=ORD_CATEGORIES,
                ),
            ),
            (
                "lr",
                LogisticRegressionMLPipeline(
                    param_grid=logistic_regression_params,
                    random_state=seed,
                    ord_categories=ORD_CATEGORIES,
                ),
            ),
            (
                "xgb",
                XGBoostMLPipeline(
                    param_grid=xgboost_params,
                    random_state=seed,
                    ord_categories=ORD_CATEGORIES,
                ),
            ),
            (
                "xgbrf",
                XGBoostRFMLPipeline(
                    param_grid=xgboost_rf_params,
                    random_state=seed,
                    ord_categories=ORD_CATEGORIES,
                ),
            ),
        ]

        # Save initial predictors (before pipeline steps) to each model folder
        initial_predictors = list(X.columns)
        exclude_str = "-".join(EXCLUDE_COL_MATCH)

        for name, pipeline in models:
            # New directory hierarchy: trainingML/{model}/seed{seed}_excl{exclude}/
            model_base_dir = os.path.join(ML_DIR, name)
            os.makedirs(model_base_dir, exist_ok=True)
            exclude_segment = f"_excl{exclude_str}" if exclude_str else ""
            run_dir = os.path.join(model_base_dir, f"seed{seed}{exclude_segment}")
            nested_dir = os.path.join(run_dir, f"nested_outer{OUTER_CV}inner{INNER_CV}")
            final_dir = os.path.join(run_dir, "final")
            os.makedirs(nested_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # --- Nested CV artifacts ---
            initial_predictors_csv = os.path.join(
                nested_dir, f"{name}_initial_predictors.csv"
            )
            pd.Series(initial_predictors, name="predictor").to_csv(
                initial_predictors_csv, index=False
            )
            print(f"Initial predictors saved to: {initial_predictors_csv}")

            print(f"\nRunning nested CV for pipeline: {name}")
            test_scores, best_params_list, cv_roc_data = nested_cv_evaluate(
                pipeline, X, y, outer_cv=OUTER_CV, inner_cv=INNER_CV, search_type="grid"
            )
            print(
                f"\n{name} mean test accuracy (nested CV): {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}"
            )

            nested_roc_plot_path = os.path.join(
                nested_dir, f"{name}_nestedcv_roc_curves.png"
            )
            nested_roc_csv_path = os.path.join(
                nested_dir, f"{name}_nestedcv_roc_data.csv"
            )
            ROCCurvePlotter.plot_nested_cv_roc_curves(
                cv_roc_data,
                save_path=nested_roc_plot_path,
                title=f"Nested CV ROC Curves: {name}",
                csv_path=nested_roc_csv_path,
            )
            print(f"Nested CV ROC curves saved to: {nested_roc_plot_path}")
            print(f"Nested CV ROC data saved to: {nested_roc_csv_path}")

            # --- Final model training with 80/20 hold-out using best hyperparameters from nested CV ---
            print(
                f"\nDeriving final hyperparameters for {name} from nested CV folds..."
            )
            # Strategy: choose the params from the fold with highest test score
            best_fold_idx = int(np.argmax(test_scores))
            chosen_params = best_params_list[best_fold_idx]
            print(f"Selected params from fold {best_fold_idx+1}: {chosen_params}")

            # 80/20 stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=seed,
                stratify=y,
            )
            print(
                f"Training final {name} model on 80% of data (hold-out 20% for unbiased evaluation)..."
            )

            # Instantiate clean pipeline of same type
            final_model = type(pipeline)(
                random_state=pipeline.random_state,
                ord_categories=pipeline.ord_categories,
            )

            # Apply chosen hyperparameters directly (except select__k if None -> 'all')
            final_model.pipeline.set_params(**chosen_params)

            # Fit on training split only
            final_model.pipeline.fit(X_train, y_train)

            from sklearn.metrics import (
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            # Compute train metrics
            y_train_pred = final_model.pipeline.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_cm = confusion_matrix(y_train, y_train_pred)
            train_cr = classification_report(y_train, y_train_pred)
            # Train specificity
            if train_cm.shape == (2, 2):
                tn, fp, fn, tp = train_cm.ravel()
                train_specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                train_specificity = np.nan
            # Train AUC
            try:
                if hasattr(final_model.pipeline.named_steps["clf"], "predict_proba"):
                    y_train_score = final_model.pipeline.predict_proba(X_train)[:, 1]
                else:
                    y_train_score = final_model.pipeline.decision_function(X_train)
                train_auc_val = roc_auc_score(y_train, y_train_score)
            except Exception:
                train_auc_val = np.nan

            # Compute test metrics
            y_test_pred = final_model.pipeline.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_cm = confusion_matrix(y_test, y_test_pred)
            test_cr = classification_report(y_test, y_test_pred)
            # Test specificity
            if test_cm.shape == (2, 2):
                tn, fp, fn, tp = test_cm.ravel()
                test_specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            else:
                test_specificity = np.nan
            # Test AUC
            if hasattr(final_model.pipeline.named_steps["clf"], "predict_proba"):
                y_test_score = final_model.pipeline.predict_proba(X_test)[:, 1]
            else:
                y_test_score = final_model.pipeline.decision_function(X_test)
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_score)
            test_auc_val = auc(test_fpr, test_tpr)

            # Store holdout ROC data for summary plot
            holdout_roc_data[name].append((test_fpr, test_tpr, test_auc_val, seed))

            # Feature importances/coefficients after fit
            if (
                hasattr(final_model.pipeline.named_steps["clf"], "feature_importances_")
                and "select" in final_model.pipeline.named_steps
            ):
                preprocessed_feature_names = final_model.get_preprocessed_feature_names(
                    final_model.pipeline, X_train
                )
                if "var_thresh" in final_model.pipeline.named_steps:
                    var_thresh_mask = final_model.pipeline.named_steps[
                        "var_thresh"
                    ].get_support()
                    features_after_var = np.array(preprocessed_feature_names)[
                        var_thresh_mask
                    ]
                else:
                    features_after_var = preprocessed_feature_names
                selected_mask = final_model.pipeline.named_steps["select"].get_support()
                selected_features = features_after_var[selected_mask]
                importances = final_model.pipeline.named_steps[
                    "clf"
                ].feature_importances_
                top_series = pd.Series(
                    importances, index=selected_features
                ).sort_values(ascending=False)
            elif (
                hasattr(final_model.pipeline.named_steps["clf"], "coef_")
                and "select" in final_model.pipeline.named_steps
            ):
                preprocessed_feature_names = final_model.get_preprocessed_feature_names(
                    final_model.pipeline, X_train
                )
                if "var_thresh" in final_model.pipeline.named_steps:
                    var_thresh_mask = final_model.pipeline.named_steps[
                        "var_thresh"
                    ].get_support()
                    features_after_var = np.array(preprocessed_feature_names)[
                        var_thresh_mask
                    ]
                else:
                    features_after_var = preprocessed_feature_names
                selected_mask = final_model.pipeline.named_steps["select"].get_support()
                selected_features = features_after_var[selected_mask]
                coefs = np.abs(final_model.pipeline.named_steps["clf"].coef_[0])
                top_series = pd.Series(coefs, index=selected_features).sort_values(
                    ascending=False
                )
            else:
                top_series = None

            final_model.metrics_ = {
                "best_params": chosen_params,
                # Train
                "train_accuracy": train_acc,
                "train_precision": precision_score(
                    y_train, y_train_pred, zero_division=0
                ),
                "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
                "train_specificity": train_specificity,
                "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
                "train_auc": train_auc_val,
                "train_confusion_matrix": train_cm,
                "train_classification_report": train_cr,
                # Test
                "test_accuracy": test_acc,
                "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
                "test_specificity": test_specificity,
                "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
                "test_auc": test_auc_val,
                "test_confusion_matrix": test_cm,
                "test_classification_report": test_cr,
            }
            if top_series is not None:
                final_model.metrics_["top_features"] = top_series
                # Record ordered list for feature usage aggregation
                feature_usage[name].append(
                    {"seed": seed, "features": top_series.index.tolist()}
                )
            else:
                # Still record placeholder to keep seed count alignment (optional)
                feature_usage[name].append({"seed": seed, "features": []})

            print(
                f"Hold-out test accuracy (20%): {test_acc:.3f} | AUC: {test_auc_val:.3f}"
            )

            # Save final model + metrics incorporating nested CV context (still reported in header)
            model_path = os.path.join(final_dir, f"{name}_final_model.pkl")
            metrics_path = os.path.join(final_dir, f"{name}_comprehensive_metrics.txt")
            final_model.save(
                model_path=model_path,
                metrics_path=metrics_path,
                nested_cv_results=(test_scores, best_params_list, name),
            )

            # Selected predictors after final training (top features if available else all)
            predictors_csv = os.path.join(final_dir, f"{name}_predictors.csv")
            if top_series is not None:
                selected_features = top_series.index.tolist()
            else:
                selected_features = list(X.columns)
            pd.Series(selected_features, name="predictor").to_csv(
                predictors_csv, index=False
            )
            print(f"Predictors saved to: {predictors_csv}")

            # Final ROC curve (hold-out test curve)
            holdout_roc_plot_path = os.path.join(
                final_dir, f"{name}_holdout_roc_curve.png"
            )
            plotter = ROCCurvePlotter(
                y_true=y_test,
                y_score=y_test_score,
                title=f"Hold-out ROC Curve: {name} (seed={seed})",
            )
            plotter.plot(show=False, save_path=holdout_roc_plot_path)
            print(f"Hold-out ROC curve saved to: {holdout_roc_plot_path}")

            # Record aggregation metrics
            aggregate_records.append(
                {
                    "model": name,
                    "seed": seed,
                    "nested_mean_acc": float(np.mean(test_scores)),
                    "nested_std_acc": float(np.std(test_scores)),
                    "holdout_accuracy": float(test_acc),
                    "holdout_precision": float(final_model.metrics_.get("test_precision", np.nan)),
                    "holdout_recall": float(final_model.metrics_.get("test_recall", np.nan)),
                    "holdout_specificity": float(final_model.metrics_.get("test_specificity", np.nan)),
                    "holdout_f1": float(final_model.metrics_.get("test_f1", np.nan)),
                    "holdout_auc": float(test_auc_val),
                    "best_params": chosen_params,
                    "num_features_top": (
                        len(top_series) if top_series is not None else 0
                    ),
                }
            )

    # After all seeds processed, plot summary ROC curves for each model
    for name, roc_data in holdout_roc_data.items():
        summary_roc_path = os.path.join(
            ML_DIR, name, f"{name}_holdout_roc_curves_across_seeds.png"
        )
        summary_roc_csv = os.path.join(
            ML_DIR, name, f"{name}_holdout_roc_data_across_seeds.csv"
        )
        ROCCurvePlotter.plot_holdout_roc_curves_across_seeds(
            roc_data,
            save_path=summary_roc_path,
            title=f"Holdout ROC Curves Across Seeds: {name}",
            csv_path=summary_roc_csv,
        )
        print(f"Summary holdout ROC curves saved to: {summary_roc_path}")
        print(f"Summary holdout ROC data saved to: {summary_roc_csv}")

    # Aggregate & write summaries
    aggregate_seed_metrics(aggregate_records, ML_DIR)
    # Aggregate feature usage across seeds per model
    aggregate_feature_usage(feature_usage, ML_DIR, top_n=None)


if __name__ == "__main__":
    main()
