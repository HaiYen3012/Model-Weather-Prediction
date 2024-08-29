import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

read_data = pd.read_csv('seattle-weather.csv')

X = read_data.drop(columns=['weather', 'date'])
Y = read_data['weather']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifierBayes = GaussianNB()
classifierLogic = LogisticRegression()
classifierTree = DecisionTreeClassifier()

classifierBayes.fit(X_train, Y_train)
classifierLogic.fit(X_train, Y_train)
classifierTree.fit(X_train, Y_train)

YpredBayes = classifierBayes.predict(X_test)
YpredLogic = classifierLogic.predict(X_test)
YpredTree = classifierTree.predict(X_test)

BayesScore = accuracy_score(Y_test, YpredBayes)
LogicScore = accuracy_score(Y_test, YpredLogic)
TreeScore = accuracy_score(Y_test, YpredTree)

print(f'Naive Bayes Accuracy Score: {BayesScore}')
print(f'Logistic Regression Accuracy Score: {LogicScore}')
print(f'Decision Tree Accuracy Score: {TreeScore}')

data_need_to_predict = [5.6, 10.0, 8.5, 3.4]
ans1 = classifierBayes.predict([data_need_to_predict])
ans2 = classifierLogic.predict([data_need_to_predict])
ans3 = classifierTree.predict([data_need_to_predict])

print(f'Naive Bayes Prediction: {ans1}')
print(f'Logistic Regression Prediction: {ans2}')
print(f'Decision Tree Prediction: {ans3}')

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['drizzle', 'rain', 'sun', 'snow', 'fog'], 
                yticklabels=['drizzle', 'rain', 'sun', 'snow', 'fog'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

cmBayes = confusion_matrix(Y_test, YpredBayes)
cmLogic = confusion_matrix(Y_test, YpredLogic)
cmTree = confusion_matrix(Y_test, YpredTree)

plot_confusion_matrix(cmBayes, title='Naive Bayes Confusion Matrix')
plot_confusion_matrix(cmLogic, title='Logistic Regression Confusion Matrix')
plot_confusion_matrix(cmTree, title='Decision Tree Confusion Matrix')

scores = {'Naive Bayes': BayesScore, 'Logistic Regression': LogicScore, 'Decision Tree': TreeScore}
plt.figure(figsize=(10, 6))
sns.barplot(x=list(scores.keys()), y=list(scores.values()))
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.title('Model Accuracy Comparison')
plt.show()