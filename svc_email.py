import pandas as pd
df = pd.read_csv('spam.csv')
print(df)

df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df)

from sklearn.feature_extraction.text import TfidfVectorizer
tfvec = TfidfVectorizer()

x = tfvec.fit_transform(df['Message'])
y = df['spam']

from sklearn.svm import SVC
svm = SVC(class_weight='balanced')

svm.fit(x,y)

new_message = 'Your bank details are being used in Asia. Is that your authorization?'
new_x = tfvec.transform([new_message])

prediction = svm.predict(new_x)[0]
print('Spam' if prediction == 1 else 'Not Spam')

from sklearn.model_selection import train_test_split as tt
x_train,x_test,y_train,y_test = tt(x,y,test_size=0.2,random_state=42)
svm.fit(x_train,y_train)
score = svm.score(x,y)
train_score = svm.score(x_train,y_train)
test_score = svm.score(x_test,y_test)
print(score)
print(train_score)
print(test_score)