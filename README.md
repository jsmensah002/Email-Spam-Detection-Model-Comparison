Project Goal:
- Compare the performance of two different email detection models and predict whether an email would be a spam or not. 

Method:
- Built and trained three separate models to predict whether an email is a spam or not.
- Compared model performance using relevant metrics, highlighting accuracy, stability, and suitability for the dataset.

Logistic Regression Model Results (with a balanced class weight):
- R^2 (Trained 80% of the dataset) = 0.9989
- R^2 (Tested 20% of the dataset) = 0.9892
- Prediction = 0 = Not Spam

Linear Support Vector Classification Model Results:
- R^2 (Trained 80% of the dataset) = 1.000
- R^2 (Tested 20% of the dataset) = 0.9919
- Prediction = 0 = Not Spam

Key Insight:
- With both models having a high test score, the final model is chosen by comparing the difference between the R² on 80% of the training data and the R² on the 20% test data. The model with the least train–test gap is preferred.
- Between logistic regression and the linear support vector classifier, the linear SVC model has the least train–test gap (0.008) and is therefore selected as the final model.
- Lower difference = less overfitting = more stable on new data.
