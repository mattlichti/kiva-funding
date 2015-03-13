import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer as lemmatizer
import filter

class FundingModel(object):

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
        
        self.vectorizer = None # vectorizer object

        self.columns = [] # list of columns in fit model
        self.tf_col = [] # list of term freq columns in fit model 
        self.model = None # maybe put the model here

    def get_themes(self, df):
        # filling null themes with emplty list so it doesn't throw errors
        df.themes = df.themes.map(lambda x: x if type(x) == list else ['none'])
        for theme in self.theme_list:
            # creating a dummy variable for each of the 22 themes
            df['theme: '+str(theme)] = df.themes.map(lambda x: theme in x)
        df = df.drop(['themes'], axis=1)
        return df

    def tokenize(self, txt, stemmer=lemmatizer()):
        return [stemmer.lemmatize(word) for word in word_tokenize(txt) if word not in [',', '.',"'s", '(', ')']]

    def disect_use(self, df):
        self.vectorizer = TfidfVectorizer(stop_words='english', tokenizer=self.tokenize, max_features=250, use_idf=False)
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

        # add currency loss and repayment interval dummies
        cur = pd.get_dummies(df.currency_loss)
        cur.columns = ["currency_loss: " + x for x in cur.columns]

        repay = pd.get_dummies(df.repayment_interval)
        repay.columns = ["repayment_interval: " + x for x in repay.columns]

        df['gender'] = df.gender == 'F'


        df = pd.concat([df, sdum, cdum, act, cur, repay], axis=1)
        df = df.drop(['posted_date', 'sector', 'use', 'country',
                      'activity', 'currency_loss', 'repayment_interval'], axis=1)
        
        print df.info()
        return df

    def fit_weighted(self, X, y, split=500, w=2, leaf=20, trees=20,
                     mf="sqrt", depth=None):
        self.model = RandomForestClassifier \
                     (min_samples_split=split, n_estimators=trees,
                      min_samples_leaf=leaf, max_features=mf, max_depth=depth)

        weights = np.array([1/(y.mean()*w) if x else 1 for x in list(y)])
        self.model.fit(X, y, sample_weight=weights)


    def transform_fit(self, df, weight=1, leaf=30, split=500,
                      trees=40, mf="sqrt", depth=None):


        use = self.disect_use(df)
        use.index = df.index

        df = self.transform(df)
        df = pd.concat([df, use], axis=1)

        y = df.pop('expired').values
        X = df.values
        self.columns = df.columns
        self.fit_weighted(X, y, w=weight, leaf=leaf, split=split, trees=trees, mf="sqrt")

    def predict (self, df):

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
        imp = self.model.feature_importances_ #only for rf

        for feat in np.argsort(imp)[::-1][0:200]:
            print str(column_list[feat]) + ": " + str(round(imp[feat] * 100,2)) + '%'

def run_model():
    df = pd.io.pickle.read_pickle('2014_df.pkl')
    df, bla = filter.get_time_periods(df)
    df.info()
    mod = FundingModel()
    mod.transform_fit(df.copy())
    mod.feat_imp()
    mod.predict(df)

if __name__ == '__main__':
    run_model()