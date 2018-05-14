
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import users
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def prepare_text(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(None, string.punctuation)
    token_dict[file] = no_punctuation

def get_vectors(X_train):
    '''
    Input:
        X_train - training set corpus of resume documents
                  or document content
        The X data is, for example, the training set of user resume content.
    Output:
        vectors - the Fit TFIDF vectors
    '''
    #this can take some time
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    #vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(X_data)

    return vectors

def multi_model_pred(X_new):
    '''
    Input:
        X_new - Either a single user resume to make a classification,
                or a test data set of documents
    Output:
        None
    '''
    # Transform the new docs to vectors using the transform vectorizer
    vecs_new = vectorizer.transform(X_new)


    models = [
        MultinomialNB(),
        SGDClassifier(),
        LinearSVC(),
        RandomForestClassifier(),
        #RandomForestRegressor(),  # <-- This wont work
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        SVC(),
        LogisticRegression()
    ]

    for model in models:

        # Fit the model with the fit_transformed tfidf vectors
        model.fit(vectors, newsgroups.target)
        # Predict using the vectors for the new docs, transformed using the
        # fit tfidf vectorizer
        y_pred = model.predict(vecs_new.todense())

        # # Use a pipeline
        # text_clf = Pipeline([('tfidf', TfidfVectorizer()),
        #                      ('clf', model)])
        # text_clf.fit(newsgroups.data, newsgroups.target)
        # # If using the pipeline, we predict using a doc dataset, not the vectorized docs
        # y_pred = text_clf.predict(docs_new)
        # # Use gridsearch to optimze params
        # from sklearn.model_selection import GridSearchCV
        # parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
        #               'tfidf__use_idf': (True, False),
        #               'clf__alpha': (1e-2, 1e-3)}
        # gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        # gs_clf = gs_clf.fit(newsgroups.data[:400], newsgroups.target[:400])
        # newsgroups.target_names[gs_clf.predict(['God is love'])[0]]
        # gs_clf.best_score_
        # for param_name in sorted(parameters.keys()):
        #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

        print model.__class__.__name__
        #print_preds(docs_new, y_pred)
        # print the % correctly predicted
        print '- score: %s' % np.mean(y_pred == news_test.target)
        from sklearn import metrics
        print metrics.classification_report(news_test.target, y_pred,
              target_names=news_test.target_names)
        print metrics.confusion_matrix(news_test.target, y_pred)
        print



def print_preds(docs_new, y_pred):
    for doc, pred in zip(docs_new, y_pred):
        print '%s: %s' % (doc, newsgroups.target_names[pred])

if __name__ == '__main__':
    pass
    #
    # # main()
    # multi_model_pred()
