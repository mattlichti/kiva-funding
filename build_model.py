import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lemmatizer


class FundingModel(object):

    def __init__(self):
        self.vect = None  # vectorizer object
        self.columns = []  # list of columns in fit model
        self.tf_col = []  # list of term freq columns in fit model
        self.model = None  # prediction model - weighted random forest

    def tokenize(self, txt, stemmer=lemmatizer()):
        '''
        tokenizes and stems text and removes stop words
        '''
        return [stemmer.lemmatize(word) for word in word_tokenize(txt)
                if word not in [',', '.', "'s", '(', ')']]

    def transform_text(self, df):
        '''
        adds frequency of 200 most common terms from the loan use text to df
        '''
        self.vect = TfidfVectorizer(stop_words='english', use_idf=False,
                                    tokenizer=self.tokenize, max_features=200)
        vect = self.vect.fit_transform(df.use.values)
        self.tf_col = ['use_%s' % x for x in self.vect.get_feature_names()]
        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        df.drop('use', axis=1, inplace=True)
        df = pd.concat([df, tf_df], axis=1)
        return df

    def transform_features(self, df):
        '''
        dummy variables for country, sector, repayment_interval, and activity
        '''
        features = ['sector', 'country', 'repayment_interval', 'activity']
        for feature in features:
            dummy = pd.get_dummies(df[feature]).astype(bool)
            dummy.columns = [feature + '_' + x.replace(' ', '_').lower()
                             for x in dummy.columns]
            df = pd.concat([df, dummy], axis=1)
        df.drop(features, axis=1, inplace=True)
        return df

    def fit_weighted_rf(self, X, y, split=400, weight=1, leaf=20, trees=40):
        self.model = RandomForestClassifier(
            min_samples_split=split, n_estimators=trees,
            min_samples_leaf=leaf, max_features='sqrt', max_depth=None)
        weights = np.array([weight/(y.mean()) if x else 1 for x in list(y)])
        self.model.fit(X, y, sample_weight=weights)

    def fit(self, df):
        '''

        '''
        df = self.transform_text(df)
        df = self.transform_features(df)
        y = df.pop('expired').values
        X = df.values
        self.columns = df.columns
        self.fit_weighted_rf(X, y)

    def predict(self, df):
        '''
        Input dataframe without lables (no expired column)
        Output array of predicted values for expiration
        '''
        vect = self.vect.transform(df.use.values)
        tf_df = pd.DataFrame(vect.toarray(), columns=self.tf_col)
        df = self.transform_features(df)
        df = pd.concat([df, tf_df], axis=1)

        new_cols = set(self.columns).difference(set(df.columns))
        del_cols = set(df.columns).difference(set(self.columns))
        df = df.drop(list(del_cols), axis=1)
        for new_col in new_cols:
            df[new_col] = 0
        return self.model.predict(df.values)

    def confusion_matrix(self, ypred, y_test):
        '''
        Returns confusion matrix and precision and recall as a string
        '''
        tp = round(np.logical_and(y_test, ypred).mean(), 3)
        fn = round(np.logical_and(y_test, ~ypred).mean(), 3)
        fp = round(np.logical_and(~y_test, ypred).mean(), 3)
        tn = round(np.logical_and(~y_test, ~ypred).mean(), 3)
        recall = round(tp / (tp + fn), 3)
        precis = round(tp / (tp + fp), 3)

        return '       Predict high risk  Predict low risk \n' + \
               'Expired: %s%%               %s%% \n' % (tp*100, fn*100) + \
               'Funded: %s%%               %s%% \n\n' % (fp*100, tn*100) + \
               'Recall: %s%%   Precision: %s%%' % (recall*100, precis*100)

    def feat_imp(self):
        '''
        Returns list of most important features as a string
        '''
        column_list = self.columns
        imp = self.model.feature_importances_
        return ''.join('%s: %s%%\n' % (column_list[feat], round(
            imp[feat] * 100, 1)) for feat in np.argsort(imp)[::-1][0:100])
