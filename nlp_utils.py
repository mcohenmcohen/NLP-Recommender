from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from itertools import product
from string import ascii_lowercase
import pandas as pd
import numpy as np
import re
import time
import shutil

GLOVE_50_PATH = '../data_sets/glove.6B.50d.txt'

# class D2V_Utils(object):
#     def __init__(self, vectorizer):
#         self.vectorizer = vectorizer

def get_doc2vec_model(in_docs):
    # Tokenize our docs
    # docs_as_tokens = get_tokenized_docs(in_docs)
    # Label the doc tokens.
    # Input can be array-like docs or list of docs as doc tokens.
    labeled_docs = get_tagged_docs(in_docs, 'job_posting')

    t1 = time.time()
    d2v_model = Doc2Vec(size=300,
                        dbow_words=1,
                        window=10,
                        min_count=5,
                        workers=4)
    print("Building Doc2Vec vocabulary...")
    t1 = time.time()
    d2v_model.build_vocab(labeled_docs)
    print("- Time: %0.3fs.\n" % (time.time() - t1))

    print("Doc2Vec model train...")
    t1 = time.time()
    d2v_model.train(labeled_docs,
                    total_examples=len(labeled_docs),
                    epochs=20)
    print("- Time: %0.3fs.\n" % (time.time() - t1))

    return d2v_model


def get_d2v_glove(infile=GLOVE_50_PATH):
    """
    https://github.com/manasRK/glove-gensim/blob/master/glove-gensim.py
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    outfile = infile.rsplit('/')[::-1][0] + '.gensim'
    # for 50 dim glove file
    line = "{} {}".format(400000, 50)
    # # For linux system, faster
    # with open(infile, 'r') as old:
    #     with open(outfile, 'w') as new:
    #         new.write(str(line) + "\n")
    #         shutil.copyfileobj(old, new)
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

    # model = Doc2Vec.load(outfile)
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(outfile)
    return model


def get_glove():
    with open(GLOVE_50_PATH, 'rb') as lines:
        w2v = {
            line.split()[0]: np.array(list(map(float, line.split()[1:]))) for line in lines
        }
    return w2v


def get_tokenized_docs(in_docs):
    '''
    Use LocalwiseVectorizer analyzer (for stop words, lemmer, regex) to
    tokenize the documents.
    '''
    # Analzye/preprocess/tokenize the words in each doc
    analyzer = LocalwiseVectorizer().build_analyzer()

    new_docs = []

    print('Tokenizing docs...')
    t1 = time.time()
    for doc in in_docs:
        if doc is None:
            continue
        # Capture words filtered by regex and stopwords.
        # Analyzer splits the doc into tokens.
        # d = [lemmer.lemmatize(w) for w in list(set(analyzer(doc)) - set(sw))
        #      if pattern.match(w)]
        d = [w for w in analyzer(doc)]
        new_docs.append(d)
    print('- Time: %s\n' % (time.time() - t1))

    return new_docs


def get_tagged_docs(in_docs, tag_prefix):
    '''
    Convert input docs to tokens, then to doc2vec TaggedDocument objects.
    If the documents are already tokens in a list object, then bypass that step

    # Doc2Vec
    # https://github.com/olafmaas/hackdelft/blob/master/hackdelft/doc2vec/train_doc2vec.py

    # Generate a TaggedDocument object from each tokenized doc
    # and append to a list
    # eg:
    #   document = TaggedDocument(words=['some', 'words', 'here'],
                                          tags=['SENT_1'])
    '''
    if type(in_docs) == list:
        in_doc_tokens = in_docs
    else:
        print('Tokenizing docs...')
        t1 = time.time()
        # in_doc_tokens = get_tokenized_docs(in_docs)
        docgen = TokenGenerator(in_docs)
        in_doc_tokens = [tokens for tokens in docgen]
        print("- Time: %0.3fs.\n" % (time.time() - t1))
    labeled_docs = []
    # print "\nCreating LabeledSentences..."
    t1 = time.time()
    for i in range(len(in_doc_tokens)):
        tag_name = '%s_%s' % (tag_prefix, i)
        labeled_doc = TaggedDocument(words=in_doc_tokens[i],
                                     tags=[tag_name])
        labeled_docs.append(labeled_doc)
    # print "- Time: %0.3fs." % (time.time() - t1)

    return labeled_docs


def vectorizer_to_dataframe(vectorizer, document_term_mat):
    '''
    Given a sklearn vectorizer return a dataframe with
    documents on rows, words on columns, weights as values.

    Input - fitted vectorizer
            document_term_mat, from vectorizer.fit_transform(docs)
    Output - pandas dataframe
    '''
    dfv = pd.DataFrame(document_term_mat.todense(),
                       columns=vectorizer.get_feature_names())
    return dfv


class LocalwiseVectorizer(TfidfVectorizer):
    '''
    http://scikit-learn.org/dev/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick
            self.vectorizer = TfidfVectorizer(token_pattern=get_token_pattern(),
                                              min_df=min_df,
                                              max_features=max_vocab_size,
                                              stop_words=get_stop_words(),
                                              ngram_range=ngram_range)
    '''
    def __init__(self, inflection_form='lemmer', max_features=5000,
                 min_df=10, max_df=1.0, ngram_range=([1, 1])):
        super(LocalwiseVectorizer, self).__init__(max_features=max_features,
                                                  min_df=min_df,
                                                  max_df=max_df,
                                                  ngram_range=ngram_range)
        self.inflection_form = inflection_form.lower()

    def build_analyzer(self):
        analyzer = super(LocalwiseVectorizer, self).build_analyzer()
        sw = get_stop_words()
        pattern = re.compile(get_token_pattern())
        if self.inflection_form == 'lemmer':
            lemmer = WordNetLemmatizer()
            return lambda doc: (lemmer.lemmatize(w) for w in list(set(analyzer(doc)) - set(sw))
                                if pattern.match(w) if len(set(w.split()).intersection(sw)) == 0)
        else:
            stemmer = PorterStemmer()
            return lambda doc: (stemmer.stem(w) for w in list(set(analyzer(doc)) - set(sw))
                                if pattern.match(w) if len(set(w.split()).intersection(sw)) == 0)


class MeanEmbeddingVectorizer(object):
    '''
    Average the vectors of each word in a document and make that
    the document vector.
    To be used with word2vec.
    '''
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
        self.analyer = LocalwiseVectorizer().build_analyzer()

    def fit(self, X, y):
        return self

    def transform(self, doc):
        '''
        Returns a document vector that is the mean of the doc's word vectors
        '''
        # prune words in doc against LW filter
        tokens = [w for w in self.analyer(doc)]
        # return the mean of words found in word2vec model (eg, GloVe)
        return np.array(
            np.mean([self.word2vec[w] for w in tokens if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
        )

class TokenGenerator:
    '''
    Generator to tokenize the documents using a vectorier analyzer
    '''
    def __init__(self, docs):
        self.docs = docs
        self.analyer = LocalwiseVectorizer().build_analyzer()

    def __iter__(self):
        for doc in self.docs:
            tokens = [w for w in self.analyer(doc)]
            yield tokens


def boost_doc_title_terms_tfidf(titles, vectorizer, document_term_mat, scale=1):
    '''
    For evary row, get the doc job title, and boost the weight for that
    title term if in the vectorizer.

    Input:
        titles - pandas series of titles strings
        vectorizer - a fitted vectorizer
        document_term_mat - captured as output of vectorizer.fit_transform
        scale - the scaling multiple for the tfidf boost for the given doc and
                title term,  capping out at 1.  Default is 1, no scaling.

    Output:
        The modified document_term_mat
    '''
    print("Boosting title terms in document term matrix...")
    t1 = time.time()

    document_term_mat = document_term_mat.tolil()  # tolil()
    all_feature_names = vectorizer.get_feature_names()

    # Add custom word stems as exceptions to the well known wordnets
    # as these are not found to be picked up by the lemmer/stemmer
    exception_stems = {
        'barista': ['baristas'],
        'busser': ['bussers']
    }
    lemmer = WordNetLemmatizer()
    sw = get_stop_words()
    pattern = re.compile(get_token_pattern())

    for doc_index in range(document_term_mat.shape[0]):
        doc_terms_indicies = document_term_mat[doc_index, :].nonzero()[1]
        doc_terms = [all_feature_names[i] for i in doc_terms_indicies]

        # doc_titles = titles.iloc[doc_index].lower().split()
        # split on any non-word character
        doc_titles = re.findall(r"[\w']+", titles.iloc[doc_index].lower())
        # filter on lemmer, alpha, pattern, and stopwords
        doc_titles = [lemmer.lemmatize(w) for w in list(set(doc_titles)-set(sw))
                      if w.isalpha()
                      if pattern.match(w)
                      if len(set(w.split()).intersection(sw)) == 0]
        for exception in exception_stems.keys():
            if exception in doc_titles:
                doc_titles = doc_titles + exception_stems[exception]
        titles_in_doc_terms = set(doc_titles).intersection(doc_terms)

        tfidf_scores = list(zip(doc_terms_indicies, [document_term_mat[doc_index, x]
                                                for x in doc_terms_indicies]))
        tfidf_dict = {all_feature_names[i]: (score, i)
                      for (i, score) in tfidf_scores}

        for t in titles_in_doc_terms:
            t_idx = tfidf_dict[t][1]
            new_tfidf = document_term_mat[doc_index, t_idx] * scale
            document_term_mat[doc_index, t_idx] = max(new_tfidf, 1.0)

    print("- Time: %0.3fs.\n" % (time.time() - t1))

    return document_term_mat.tocsr()


def get_token_pattern():
    token_pattern = '(?ui)\\b[a-zA-Z]*[a-z]+\\w*\\b'
    return token_pattern


def get_stop_words():
    cusotm_stop_words = \
       ['http', 'www', 'nbsp', 'com', 'org', 'edu', 'mailto', 'html',
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'll', 'am', 'pm', 'minutes', 'hours', 'time', 'reasonable',
        'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'days', 'week', 'saturday', 'sunday', 'weekday', 'weekend', 'month',
        'year', 'work', 'working', 'available', 'availability', 'team', 'plus',
        'position', 'join', 'job', 'jobs', 'duties', 'looking', 'able',
        'excellent', 'good', 'great', 'strong', 'flexible', 'accomplish',
        'ability', 'requirement', 'great', 'resume', 'love', 'like', 'high',
        'experience', 'experienced', 'skill', 'include', 'including',
        'qualified', 'application', 'applicants', 'apply', 'individuals',
        'san', 'francisco', 'chicago', 'oakland', 'alameda', 'napa',
        'street', 'blvd', 'st', 'ave', 'inter', 'drop', 'skills', 'skilled',
        'people', 'person', 'pay', 'company', 'need', 'shifts', 'interested',
        'make', 'email', 'want', 'send', 'friendly', 'staff', 'offer', 'seeking',
        'candidate', 'professional', 'communication', 'qualification', 'group',
        'health', 'benefit', 'attitude', 'positive', 'new', 'friendly',
        'requirement', 'minimum', 'preferred', 'start', 'self', 'fun',
        'goal', 'help', 'knowledge', 'best', 'learn', 'employee', 'opportunity',
        'location', 'training', 'benefit', 'responsibility', 'order', 'maintain',
        'disability', 'race', 'origin', 'sex', 'sexual', 'party', 'orientation',
        'arrest', 'conviction', 'marital', 'national', 'ordinance', 'color',
        'effectuate', 'creed', 'proper', 'benefit', 'dental', 'paid', 'medical',
        'vacation', 'competitive', 'insurance', 'sick', 'holiday', 'vision',
        'bonus', 'discount', 'plan', 'compensation', 'wage', 'rate', 'earn',
        'long', 'term', 'environment', 'employment', 'come', 'leave', 'punctual',
        'bring', 'evening', 'night', 'berkeley', 'shattuck', 'hiring', 'avenue',
        'blvd', 'boulevard', 'place', 'note', 'palo', 'alto', 'thoroughly',
        'needed', 'enjoy', 'essy', 'text', 'energy', 'worked', 'reliable',
        'sacramento', 'hard', 'big', 'bigger', 'south', 'north', 'east', 'west',
        'rising', 'wendy', 'brucker', 'featured', 'solano', 'owner', 'peninsula',
        'locate', 'introducing', 'regional', 'bay', 'area', 'free', 'city',
        'valley', 'located', 'hallway', 'arriving', 'decide', 'tap', 'started',
        'asdf', 'persuasion', 'select', 'extent', 'strictly', 'admitted',
        'treated', 'discrimination', 'religious', 'relating', 'staffing', 'fair',
        'character', 'discus', 'upstairs', 'super', 'motivate', 'type',
        'required',

        'episode', 'unspecified', 'specified', 'stated', 'condition', 'care',
        'not', 'applicable', 'condition', 'effect', 'classified', 'personal',
        'involving', 'unknown', 'degree'
        ]

    # Remove all 2 letter words
    all_two_letter_words = list(map(''.join, product(ascii_lowercase, repeat=2)))

    added_stop_words = cusotm_stop_words + all_two_letter_words

    stop_words = text.ENGLISH_STOP_WORDS.union(added_stop_words)

    return stop_words
