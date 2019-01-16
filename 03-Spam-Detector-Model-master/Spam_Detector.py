import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import _pickle as c

def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print ("saved")

df = pd.read_csv('SMSSpamCollection',sep='\t',names=['Status','Message'])

df.loc[df['Status'] == 'ham','Status'] = 1
df.loc[df['Status'] == 'spam','Status'] = 0


df_x = df['Message']
df_y = df['Status']

x_train, x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2)

cv = TfidfVectorizer(min_df=1,stop_words='english')

x_traincv = cv.fit_transform(x_train)
x_testcv = cv.transform(x_test)

a = x_traincv.toarray()

mnb = MultinomialNB()

y_train = y_train.astype('int')

mnb.fit(x_traincv,y_train)


pred = mnb.predict(x_testcv)

actual = np.array(y_test)

count = 1

for i in range(len(pred)):
    if pred[i] == actual[i]:
        count += 1
print("Accurecy :{}".format(count/len(pred)))
save(mnb,'Spam_Detector_Model.mdl')