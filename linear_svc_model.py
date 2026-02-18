from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

X = df['Message']
y = df['spam']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + LinearSVC
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', LinearSVC(max_iter=5000))  # increase max_iter to ensure convergence
])

# Grid search parameters
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__class_weight': [None, 'balanced'],
    'svc__tol': [1e-3, 1e-4]  # optional tuning
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Train/test scores
train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)
print("Best Parameters:", grid.best_params_)
print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")

# Save the full pipeline
joblib.dump(best_model, 'spam_model_pipeline_svc.pkl')
print("Pipeline saved as spam_model_pipeline_svc.pkl")