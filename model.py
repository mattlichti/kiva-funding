import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lemmatizer


class funding_model(object):

    def __init__(self):
        self.theme_list = sorted(['Underfunded Areas', 'Rural Exclusion',
                                  'Vulnerable Groups', 'Conflict Zones',
                                  'Mobile Technology', 'Green',
                                  'Higher Education', 'Start-Up',
                                  'Arab Youth', 'SME',
                                  'Water and Sanitation', 'Youth',
                                  'Islamic Finance', 'Job Creation',
                                  'Fair Trade', 'Kiva City LA',
                                  'Innovative Loans', 'Growing Businesses',
                                  'Disaster recovery',
                                  'Health', 'Kiva City Detroit',
                                  'Flexible Credit Study', 'none'])

        self.columns = []

    def get_themes(self, df):
        # filling null themes with emplty list so it doesn't throw errors
        df.themes = df.themes.map(lambda x: x if type(x) == list else ['none'])
        for theme in self.theme_list:
            # creating a dummy variable for each of the 22 themes
            df['theme: '+str(theme)] = df.themes.map(lambda x: theme in x)
        df = df.drop(['themes'], axis=1)
        return df

    def tokenize(self, txt, stemmer=lemmatizer()):
        return [stemmer.lemmatize(word) for word in word_tokenize(txt) if word not in [',', '.',"'s"]]

    def disect_use(self, df):
        self.vectorizer = TfidfVectorizer(stop_words='english', tokenizer=self.tokenize, max_features=200, smooth_idf=True)
        vect = self.vectorizer.fit_transform(df.use.values)
        self.tf_col = ['use: ' + str(x) for x in self.vectorizer.get_feature_names()]
        uses = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        return uses

    def transform(self, df):
        df = self.get_themes(df)

        # adding activity, sector, country dummies
        acts = df.activity.map(lambda x: "activity: " + x)
        act = pd.get_dummies(acts)
        sdum = pd.get_dummies(df.sector)
        sdum.columns = ['sector: ' + x for x in sdum.columns]
        cdum = pd.get_dummies(df.country)
        cdum.columns = ['country: ' + x for x in cdum.columns]

        df['monthly'] = ~(df.pay_at_end | df.irregular_payments)
        df = pd.concat([df, sdum, cdum, act], axis=1)
        df = df.drop(['posted_date', 'sector', 'tags', 'use', 'country',
                      'activity'], axis=1)
        return df

    def fit_oversample(self, X, y, ratio=1, split=4):
        X_expired, y_expired = X[y], y[y]
        X_funded, y_funded = X[~y], y[~y]
        funded_prop = y_expired.shape[0]/float(y_funded.shape[0])
        X_sample, X_junk, y_sample, y_junk = train_test_split(
            X_funded, y_funded, train_size=funded_prop*ratio)

        X_oversamp = np.concatenate([X_expired, X_sample], axis=0)
        y_oversamp = np.concatenate([y_expired, y_sample], axis=0)
        self.model = RandomForestClassifier(n_estimators=15,
                                            min_samples_split=split)
        self.model.fit(X_oversamp, y_oversamp)

    def fit_weighted(self, X, y, split=500, w=2, leaf=20, trees=20,
                     mf="sqrt", depth=None):
        self.model = RandomForestClassifier \
                     (min_samples_split=split, n_estimators=trees,
                      min_samples_leaf=leaf, max_features=mf, max_depth=depth)

        weights = np.array([1/(y.mean()*w) if x else 1 for x in list(y)])
        self.model.fit(X, y, sample_weight=weights)

    def fit_svm(self, X, y, class_weight='auto'):
            # weights = y/(y.mean()/w) + 1
            self.model = SVC(class_weight='auto')
            self.model.fit(X, y)

    def transform_fit(self, df, mod='weighted', weight=2, leaf=20, split=500,
                      trees=20, mf="sqrt", depth=None):
        df = df.copy()
        y = df.pop('expired').values

        use = self.disect_use(df)
        use.index = df.index

        df = self.transform(df)
        df = pd.concat([df, use], axis=1)

        X = df.values
        self.columns = df.columns
        if mod == 'weighted':
            self.fit_weighted(X, y, w=weight, leaf=leaf, split=split, trees=trees, mf="sqrt")
        elif mod == 'SVM':
            self.fit_svm(X, y)
        else:
            self.fit_oversample(X, y)

    def predict (self, df):
        df = df.copy()
        y = df.pop('expired').values

        vect = self.vectorizer.transform(df.use.values)
        uses = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        uses.index = df.index

        df = self.transform(df)
        df = pd.concat([df, uses], axis=1)

        df_cols = set(df.columns)
        fit_cols = set(self.columns)
        new_cols = fit_cols.difference(df_cols)
        del_cols = df_cols.difference(fit_cols) 
        df = df.drop(list(del_cols),axis=1)

        for new_col in new_cols:
            df[new_col] = 0
        X = df.values
        ypred = self.model.predict(X)
        self.confusion(ypred, y)

    def confusion(self, ypred, y_test):
        true_pos = np.logical_and(y_test, ypred).mean()
        false_pos = np.logical_and(~y_test, ypred).mean()
        true_neg = np.logical_and(~y_test, ~ypred).mean()
        false_neg = np.logical_and(y_test, ~ypred).mean()
        recall = true_pos / (true_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        print 'actl pos:', round(y_test.mean(), 4), 'pred pos:', round(ypred.mean(), 4)
        print 'true pos:', round(true_pos, 4), 'false pos:', round(false_pos, 4)
        print 'true neg:', round(true_neg, 4), 'false neg:', round(false_neg, 4)
        print 'recall:', round(recall, 4), 'precision:', round(precision, 4)
        print ''

    def feat_imp(self):
        column_list = self.columns
        imp = self.model.feature_importances_
        most = np.argsort(imp)[::-1]
        # most = most[::-1]
        for feat in most:
            print str(column_list[feat]) + ": " + str(round(imp[feat] * 100,2)) + '%'


def run_model():
    df = pipe.load_cleaned(['timeframe'])
    mod = funding_model()
    mod.transform_fit(df)
    mod.feat_imp()

if __name__ == '__main__':
    run_model()