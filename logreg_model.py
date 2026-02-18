from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv('spam.csv')
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

x = df['Message']
y = df['spam']

# Split train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + LogisticRegression
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(max_iter=1000, solver='saga'))
])

# Grid search
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__class_weight': [None, 'balanced']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x_train, y_train)

best_model = grid.best_estimator_

# Train/test scores
train_score = best_model.score(x_train, y_train)
test_score = best_model.score(x_test, y_test)
print("Best Parameters:", grid.best_params_)
print(f"Train Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")

# Save full pipeline
joblib.dump(best_model, 'spam_model_pipeline.pkl')
