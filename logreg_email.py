#Importing file using pandas
import pandas as pd
df = pd.read_csv('spam.csv')
print(df)

#Adding a new column to convert spams to 1 and hams to 0
df['spam'] = df['Category'].apply(lambda x: 1 if x =='spam' else 0)
print(df)

#Converting message column to bag of words using CountVectorizer and defining x and y
from sklearn.feature_extraction.text import TfidfVectorizer
tfvec = TfidfVectorizer()

x = tfvec.fit_transform(df['Message'])
y = df['spam']

#Import logistic regression
import sklearn.linear_model as linear_model
logreg = linear_model.LogisticRegression(class_weight='balanced', max_iter=1000)
logreg.fit(x,y)

#prediction test
new_message = 'We are sorry to announce that your account has been locked. Kindly send your password for us to restore'
new_x = tfvec.transform([new_message])

prediction = logreg.predict(new_x)[0]
print('Spam' if prediction == 1 else 'Not Spam')

#using train test split 
from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)
logreg.fit(x_train,y_train)
score = logreg.score(x,y)
train_score = logreg.score(x_train,y_train)
test_score = logreg.score(x_test,y_test)
print(score)
print(train_score)
print(test_score)


