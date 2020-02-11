# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
df=pd.read_csv(path)
print(df.iloc[:,:7])
X=df.drop(['attr5'],axis=1)
y=df['attr5'].copy()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)
scalar=MinMaxScaler()
scalar.fit(X_train)
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)
# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
roc_score=roc_auc_score(y_test,y_pred)
print(roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=4)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
roc_score=roc_auc_score(y_test,y_pred)
print('Score is:',roc_score)



# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here
rfc=RandomForestClassifier(random_state=4)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
roc_score=roc_auc_score(y_test,y_pred)
print('Score is:',roc_score)
# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here
bagging_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=100,random_state=0)
bagging_clf.fit(X_train,y_train)
score_bagging=bagging_clf.score(X_test,y_test)
print('Bagging score:',score_bagging)
# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
voting_clf_hard=VotingClassifier(estimators=model_list,voting='hard')
voting_clf_hard.fit(X_train,y_train)
hard_voting_score=voting_clf_hard.score(X_test,y_test)
print('hard voting score:',hard_voting_score)
# Code ends here


