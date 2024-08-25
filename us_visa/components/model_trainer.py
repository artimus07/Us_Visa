import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_numpy_array_data, load_object, save_object
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa.entity.estimator import USvisaModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initializes the ModelTrainer class with the required configuration and artifact objects.
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the transformed training and testing data.
        
        :return: A tuple of training and testing data arrays.
        """
        try:
            logging.info("Loading transformed training and testing data.")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            if train_arr is None or test_arr is None:
                raise USvisaException("Failed to load training or testing data.", sys)

            return train_arr, test_arr
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _extract_features_and_labels(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features and labels from the data array.
        
        :param data: The data array.
        :return: A tuple of features and labels arrays.
        """
        try:
            logging.info("Extracting features and labels.")
            X = data[:, :-1]
            y = data[:, -1]
            return X, y
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _get_best_model(self, X_train: np.ndarray, y_train: np.ndarray) -> object:
        """
        Finds the best model using GridSearchCV with RandomForest and SVC.
        
        :param X_train: The feature set for training.
        :param y_train: The target labels for training.
        :return: The best model object.
        """
        try:
            logging.info("Finding the best model using GridSearchCV.")
            
            # Define models and their hyperparameters
            model_params = {
                'random_forest': {
                    'model': RandomForestClassifier(),
                    'params': {
                        'n_estimators': [10, 50, 100],
                        'max_depth': [None, 10, 20, 30],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'svc': {
                    'model': SVC(probability=True),
                    'params': {
                        'C': [0.1, 1, 10],
                        'kernel': ['linear', 'rbf'],
                        'gamma': ['scale', 'auto']
                    }
                }
            }

            # Perform grid search for each model
            best_score = 0
            best_model = None
            for model_name, model_info in model_params.items():
                grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_

            if best_score < self.model_trainer_config.expected_accuracy:
                raise USvisaException("No suitable model found that meets the expected accuracy.", sys)

            return best_model
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _evaluate_model(self, model_obj: object, X_test: np.ndarray, y_test: np.ndarray) -> ClassificationMetricArtifact:
        """
        Evaluates the model on the test data and returns the classification metrics.
        
        :param model_obj: The trained model object.
        :param X_test: The feature set for testing.
        :param y_test: The target labels for testing.
        :return: A ClassificationMetricArtifact object containing the evaluation metrics.
        """
        try:
            logging.info("Evaluating the model on the test data.")
            y_pred = model_obj.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            return ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def _save_model(self, model_obj: object, preprocessing_obj: object) -> None:
        """
        Saves the trained model along with the preprocessing object.
        
        :param model_obj: The trained model object.
        :param preprocessing_obj: The preprocessing object used during training.
        """
        try:
            logging.info("Saving the trained model and preprocessing object.")
            usvisa_model = USvisaModel(preprocessing_object=preprocessing_obj, trained_model_object=model_obj)
            save_object(self.model_trainer_config.trained_model_file_path, usvisa_model)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates the model training process and returns the resulting artifact.
        
        :return: A ModelTrainerArtifact containing the trained model and associated metrics.
        """
        logging.info("Initiating the model training process.")
        try:
            # Load data
            train_arr, test_arr = self._load_data()

            # Extract features and labels
            X_train, y_train = self._extract_features_and_labels(train_arr)
            X_test, y_test = self._extract_features_and_labels(test_arr)

            # Get the best model
            best_model = self._get_best_model(X_train, y_train)
            if best_model is None:
                raise USvisaException("No suitable model found that meets the expected accuracy.", sys)

            # Evaluate the model
            metric_artifact = self._evaluate_model(best_model, X_test, y_test)

            # Load the preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            if preprocessing_obj is None:
                raise USvisaException("Failed to load the preprocessing object.", sys)

            # Save the model
            self._save_model(best_model, preprocessing_obj)

            # Return the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model training process completed successfully: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            logging.error("Error occurred during the model training process.")
            raise USvisaException(e, sys) from e
