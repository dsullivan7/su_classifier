import json
import csv
import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier 

try:
    os.remove('out.csv')
except OSError:
    pass

with open('train.tsv','rb') as train_in, open('test.tsv','rb') as test_in, open('out.csv', 'wb') as csvout:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    train_in = csv.reader(train_in, delimiter='\t')
    test_in = csv.reader(test_in, delimiter='\t')
    csvout = csv.writer(csvout)
    header_train = train_in.next()
    header_test = test_in.next()

    csvout.writerow(['urlid','label'])
    body_index = header_train.index('boilerplate')
    label_index = header_train.index('label')
    urlid_index = header_test.index('urlid')
    indices = []
    #indices.append(header_test.index('alchemy_category_score'))
    indices.append(header_test.index('commonlinkratio_1'))
    indices.append(header_test.index('commonlinkratio_2'))
    indices.append(header_test.index('commonlinkratio_3'))
    indices.append(header_test.index('commonlinkratio_4'))
    indices.append(header_test.index('hasDomainLink'))
    indices.append(header_test.index('html_ratio'))
    indices.append(header_test.index('image_ratio'))
    #indices.append(header_test.index('is_news'))
    indices.append(header_test.index('lengthyLinkDomain'))
    indices.append(header_test.index('linkwordscore'))
    #indices.append(header_test.index('news_front_page'))
    indices.append(header_test.index('numberOfLinks'))

    docs = []
    labels = []
    for row in train_in:
        vect = []
        for i in indices:
            vect.append(float(row[i]))
        body = (json.loads(row[body_index]))['body']
        label = int(row[label_index])
        """
        if not body is None:
            docs.append(body)
            labels.append(label)
        """
        docs.append(vect)
        labels.append(label)

    #X_train = vectorizer.fit_transform(docs)
   
    clf1 = RidgeClassifier(tol=1e-2, solver="lsqr") 
    clf2 = BernoulliNB(alpha=.01)
    clf3 = Perceptron(n_iter=50)
    clf1.fit(docs, labels)
    clf2.fit(docs, labels)
    clf3.fit(docs, labels)

    docs = []
    ids = []
    for row in test_in:
        vect = []
        for i in indices:
            vect.append(float(row[i]))
        docs.append(vect)
        ids.append(row[urlid_index])
        body =  (json.loads(row[body_index]))['body']
        """
        if not body is None:
            docs.append(body)
        else:
            docs.append("null")
        """
    #docs = vectorizer.transform(docs)
    predictions1 = clf1.predict(docs);
    predictions2 = clf2.predict(docs);
    predictions3 = clf3.predict(docs);
    for idx,id in enumerate(ids):
        answer =  predictions2[idx] 
        csvout.writerow([id, int(answer)])
