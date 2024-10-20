# Titanic Survival Prediction Project

## Overview

This project focuses on building and comparing multiple **predictive models** to forecast the likelihood of survival for individual passengers aboard the RMS Titanic. In addition to predicting survival outcomes, the project also aims to identify key **demographic characteristics** that are associated with a higher probability of survival.

The project involves a comprehensive comparison of different machine learning models through cross-validation and hyperparameter tuning. The goal is to evaluate the models based on their performance and select the best one for accurate survival prediction.

## Project Goals

- **Predict Survival**: Develop models to estimate the likelihood of survival for Titanic passengers.
- **Analyze Demographics**: Identify demographic features (e.g., age, gender, class) that influence survival rates.
- **Model Comparison**: Compare the performance of different models using cross-validation and fine-tune their hyperparameters.
- **Maximize Accuracy**: Use the **accuracy metric** to evaluate and select the best model based on prediction performance.

## Models Used

The project compares the following five machine learning models:
1. **Logistic Regression**: A basic yet effective linear model for binary classification.
2. **Random Forest**: An ensemble learning method based on decision trees that improves accuracy by reducing overfitting.
3. **Decision Trees**: A tree-based model that splits data based on features to make predictions.
4. **Support Vector Machine (SVM)**: A model that separates classes by finding the optimal hyperplane.
5. **Neural Networks**: A simple feedforward neural network with one hidden layer, used to model complex patterns.

Each model undergoes hyperparameter tuning and cross-validation to find the best configurations before being compared based on test accuracy.

## Project Structure

1. **Data Preprocessing**:
   - Handling missing values in the dataset.
   - Encoding categorical variables such as gender and embarked port.
   - Scaling numerical features like age and fare to ensure better model performance.

2. **Model Building & Cross-Validation**:
   - Each model is trained using **cross-validation** to avoid overfitting and to assess performance across different data splits.
   - **Hyperparameter tuning** is performed using grid search or random search to find the best-performing configurations for each model.

3. **Model Comparison**:
   - After tuning, the models are compared based on their **cross-validated accuracy** and **test set performance**.
   - The model with the highest accuracy is selected as the best predictor for survival on the Titanic.

4. **Feature Importance**:
   - Analyze the importance of different features (e.g., age, sex, class) in determining survival, particularly with tree-based models and logistic regression.

## Dataset

The Titanic dataset contains the following features:
- **Pclass**: Passenger’s socio-economic status (1st, 2nd, 3rd class).
- **Sex**: Gender of the passenger.
- **Age**: Passenger’s age.
- **SibSp**: Number of siblings or spouses aboard the Titanic.
- **Parch**: Number of parents or children aboard.
- **Fare**: The amount paid for the ticket.
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

These features, along with others, are used to predict survival.

## How to Run the Project

1. **Install Dependencies**:
   Ensure you have Python installed along with required libraries like `scikit-learn`, `pandas`, `numpy`, and `matplotlib`:
   ```bash
   pip install scikit-learn pandas numpy matplotlib
   ```

2. **Run the Code**:
   To compare the models and view the results, run the provided Python scripts or notebook:
   ```bash
   python titanic_classification.py
   ```

3. **View Model Results**:
   The script will output the performance of each model and display the best-performing model based on cross-validation accuracy.

## Conclusion

This project evaluates multiple machine learning models to predict survival on the Titanic, with a focus on both model accuracy and feature importance. By comparing models such as Logistic Regression, Random Forest, Decision Trees, SVM, and Neural Networks, the project identifies the best model to maximize prediction accuracy.

---

Feel free to explore the code and simulations, and for any contributions or questions, you can use the repository's issue tracker.
