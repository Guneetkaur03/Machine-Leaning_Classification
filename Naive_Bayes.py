from sklearn.datasets import fetch_20newsgroups  //import the Dataset from sklearn 
newsgroups_train = fetch_20newsgroups(subset='train') //training Data

categ = ['sci.space','sci.med','rec.sport.hockey','alt.atheism','comp.graphics'] //Define the categories that you want to Select
newsgroups_train = fetch_20newsgroups(subset='train', categories=categ, random_state =0) //Training Data which contains only those categories.
newsgroups_train.target_names //prints the categories


from sklearn.feature_extraction.text import CountVectorizer //Collection of text converted to Count
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(newsgroups_train.data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer //Transform Count to term Frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, newsgroups_train.target)

docs_new = ['Heart Cancer is diagnosed','celestial bodies in galaxy','Chak de India movie was based on Hockey','nasa rocket launcher','rockets','guru nanak dev god of sikhs','GPU are faster than CPU']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, newsgroups_train.target_names[category]))

newsgroups_test = fetch_20newsgroups(subset='test',categories=categ, random_state=0)
docs_test = newsgroups_test.data
X_counts = count_vect.transform(docs_test)
X_tfidf = tfidf_transformer.transform(X_counts)
new_predicted = clf.predict(X_tfidf)
from sklearn import metrics
print(metrics.classification_report(newsgroups_test.target, new_predicted, target_names=newsgroups_test.target_names)) //prints the precision and F1 score

