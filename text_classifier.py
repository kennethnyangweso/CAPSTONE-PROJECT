import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union

from sklearn.base import BaseEstimator
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


class TextClassifier:
    """A class for training and evaluating multiple text classification models.
    
    This version assumes that data is preprocessed and vectorized before being passed in.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the TextClassifier.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}             # Dictionary to store model instances
        self.trained_models = {}     # Dictionary to store trained model instances
        self.best_model_name = None  # Track the best model by name
        self.best_score = 0.0        # Store the best F1 score
        self.category_names = None   # Optional list of class names for reporting
        self.feature_names = None    # Optional list of feature names for feature importance

    def set_category_names(self, categories: List[str]) -> None:
        """Set the category names for evaluation reports.
        
        Args:
            categories: List of class names.
        """
        self.category_names = categories

    def set_feature_names(self, features: List[str]) -> None:
        """Set the feature names for feature importance analysis.
        
        Args:
            features: List of feature names.
        """
        self.feature_names = features

    def add_model(self, name: str, model: BaseEstimator) -> None:
        """Add a model to the classifier.
        
        Args:
            name: Name of the model.
            model: Scikit-learn estimator.
        """
        self.models[name] = model

    def add_default_models(self) -> None:
        """Add a set of default classification models."""
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state),
            "Linear SVM": LinearSVC(class_weight='balanced', random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "Neural Network": MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50), random_state=self.random_state)
        }

    def create_ensemble(self, model_names: List[str] = None, voting: str = 'soft') -> None:
        """Create an ensemble model from existing models.
        
        Args:
            model_names: List of model names to include in ensemble (uses all if None).
            voting: Type of voting ('hard' or 'soft').
        """
        if not model_names:
            model_names = list(self.models.keys())
        
        estimators = [(name, self.models[name]) for name in model_names 
                      if name in self.models and hasattr(self.models[name], "predict_proba")]
        
        if len(estimators) < 2:
            raise ValueError("Need at least 2 models with predict_proba for ensemble")
        
        ensemble = VotingClassifier(estimators=estimators, voting=voting)
        self.models["Ensemble"] = ensemble

    def handle_imbalance(self, X_train: np.ndarray, y_train: np.ndarray, 
                         method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance in the training data.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            method: Method for handling imbalance ('smote').
            
        Returns:
            Tuple of resampled (X_train, y_train).
        """
        if method.lower() == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            return X_train_resampled, y_train_resampled
        else:
            return X_train, y_train

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                         handle_imbalance: bool = False) -> Dict:
        """Train all added models on the given data.
        
        Args:
            X_train: Preprocessed training features.
            y_train: Preprocessed training labels.
            handle_imbalance: Whether to apply imbalance handling.
            
        Returns:
            Dictionary of trained models.
        """
        if handle_imbalance:
            X_train, y_train = self.handle_imbalance(X_train, y_train)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        
        return self.trained_models

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray, 
                      detailed: bool = True) -> Dict:
        """Evaluate a single model.
        
        Args:
            model_name: Name of the model to evaluate.
            X_test: Preprocessed test features.
            y_test: Preprocessed test labels.
            detailed: Whether to return detailed metrics.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        if detailed:
            # Use category names if provided, otherwise omit target_names
            if self.category_names:
                results['classification_report'] = classification_report(
                    y_test, y_pred, target_names=self.category_names, output_dict=True
                )
                results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            else:
                results['classification_report'] = classification_report(
                    y_test, y_pred, output_dict=True
                )
                results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        return results

    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                           detailed: bool = True) -> Dict:
        """Evaluate all trained models.
        
        Args:
            X_test: Preprocessed test features.
            y_test: Preprocessed test labels.
            detailed: Whether to return detailed metrics.
            
        Returns:
            Dictionary of evaluation results for all models.
        """
        results = {}
        self.best_score = 0.0
        self.best_model_name = None
        
        for name in self.trained_models:
            print(f"Evaluating {name}...")
            model_results = self.evaluate_model(name, X_test, y_test, detailed)
            results[name] = model_results
            
            # Track best model based on F1 score
            if model_results['f1_score'] > self.best_score:
                self.best_score = model_results['f1_score']
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with F1 score: {self.best_score:.4f}")
        return results

    def cross_validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5, metrics: List[str] = None) -> Dict:
        """Perform cross-validation for a single model.
        
        Args:
            model_name: Name of the model to cross-validate.
            X: Preprocessed feature matrix.
            y: Preprocessed target labels.
            cv: Number of cross-validation folds.
            metrics: List of metrics to compute.
            
        Returns:
            Dictionary of cross-validation results.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if metrics is None:
            metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        results = {}
        for metric in metrics:
            scores = cross_val_score(
                self.models[model_name], X, y, cv=cv, scoring=metric
            )
            results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        return results

    def tune_hyperparameters(self, model_name: str, param_grid: Dict, 
                            X_train: np.ndarray, y_train: np.ndarray,
                            cv: int = 5, scoring: str = 'f1_macro') -> BaseEstimator:
        """Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to tune.
            param_grid: Grid of parameters to search.
            X_train: Preprocessed training features.
            y_train: Preprocessed training labels.
            cv: Number of cross-validation folds.
            scoring: Metric to optimize.
            
        Returns:
            Tuned model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best {scoring} score: {grid_search.best_score_:.4f}")
        
        # Update the model with the best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_

    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a trained model.
        
        Args:
            model_name: Name of the model.
            top_n: Number of top features to return.
            
        Returns:
            DataFrame of feature importances.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # For linear models that have coef_ attribute
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} doesn't support feature importance")
        
        if not self.feature_names:
            raise ValueError("Feature names are not set. Use set_feature_names() before calling this method.")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)

    def plot_confusion_matrix(self, model_name: str, results: Dict = None,
                             X_test: np.ndarray = None, y_test: np.ndarray = None,
                             figsize: Tuple[int, int] = (10, 8),
                             cmap: str = 'Blues') -> None:
        """Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model.
            results: Pre-computed evaluation results (optional).
            X_test: Preprocessed test features (used if results not provided).
            y_test: Preprocessed test labels (used if results not provided).
            figsize: Figure size.
            cmap: Color map for the plot.
        """
        if results is None and (X_test is None or y_test is None):
            raise ValueError("Either results or both X_test and y_test must be provided")
        
        if results is not None and model_name in results:
            cm = results[model_name]['confusion_matrix']
        else:
            y_pred = self.trained_models[model_name].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        # If category_names is set, use it; otherwise, use default indices.
        xticklabels = self.category_names if self.category_names else None
        yticklabels = self.category_names if self.category_names else None
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=xticklabels, yticklabels=yticklabels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self, model_name: str, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, train_sizes: np.ndarray = None,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot learning curve for a model.
        
        Args:
            model_name: Name of the model.
            X: Preprocessed feature matrix.
            y: Preprocessed target labels.
            cv: Number of cross-validation folds.
            train_sizes: Array of training size proportions.
            figsize: Figure size.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, test_scores = learning_curve(
            self.models[model_name], X, y, cv=cv, 
            train_sizes=train_sizes, scoring='accuracy'
        )
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Training score')
        plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Cross-validation score')
        plt.title(f'Learning Curve - {model_name}')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_errors(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      X_test_text: List[str], max_examples: int = 10) -> pd.DataFrame:
        """Analyze misclassified examples.
        
        Args:
            model_name: Name of the model.
            X_test: Preprocessed test features.
            y_test: Preprocessed test labels.
            X_test_text: Original text of test examples.
            max_examples: Maximum number of error examples to return.
            
        Returns:
            DataFrame of misclassified examples.
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        # Find misclassified examples
        errors_idx = np.where(y_pred != y_test)[0]
        
        error_data = []
        for i in errors_idx[:max_examples]:
            error_data.append({
                'text': X_test_text[i],
                'true_label': self.category_names[y_test[i]] if self.category_names else y_test[i],
                'predicted_label': self.category_names[y_pred[i]] if self.category_names else y_pred[i]
            })
        
        return pd.DataFrame(error_data)

    def save_model(self, model_name: str, filepath: str) -> None:
        """Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save.
            filepath: Path to save the model.
        """
        import joblib
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)

    def load_model(self, model_name: str, filepath: str) -> BaseEstimator:
        """Load a trained model from disk.
        
        Args:
            model_name: Name to associate with the loaded model.
            filepath: Path to the saved model.
            
        Returns:
            Loaded model.
        """
        import joblib
        
        model = joblib.load(filepath)
        self.models[model_name] = model
        self.trained_models[model_name] = model
        
        return model
