Project Goal:
- Compare the performance of three different email detection models and predict whether an email would be a spam or not. 

Method:
- Built and trained three separate models to predict whether an email is a spam or not.
- Compared model performance using relevant metrics, highlighting accuracy, stability, and suitability for the dataset.

Logistic Regression Model Results (with a balanced class weight):
- R^2 (full dataset) = 0.9903
- R^2 (Trained 80% of the dataset) = 0.9921
- R^2 (Tested 20% of the dataset) = 0.9830
- Prediction = 0 = Not Spam

Support Vector Classification Model Results:
- R^2 (full dataset) = 0.9964
- R^2 (Trained 80% of the dataset) = 0.9998
- R^2 (Tested 20% of the dataset) = 0.9830
- Prediction = 0 = Not Spam

Random Forest Classifier Model Results:
- R^2 (full dataset) = 0.9962
- R^2 (Trained 80% of the dataset) = 1.0000
- R^2 (Tested 20% of the dataset) = 0.9812
- Prediction = 0 = Not Spam

Key Insight:
- Based on the test performance, both the logistic regression model and the support vector classifier achieved the same test score.
- The final model now is chosen by comparing the difference between the R² on 80% of the training data and the R² on the 20% test data. The model with the smaller difference is preferred.
- Between logistic regression and the support vector classifier, logistic regression has the smaller difference (0.0092) and is therefore selected as the final model.
- Lower difference = less overfitting = more stable on new data.
