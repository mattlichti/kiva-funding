import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import balance_weights


class funding_model(object):

	def __init__(self):
		self.theme_list = sorted([u'Underfunded Areas', u'Rural Exclusion',
		u'Vulnerable Groups', u'Conflict Zones', u'Mobile Technology',
		u'Green', u'Higher Education', u'Start-Up', u'Arab Youth', u'SME',
		u'Water and Sanitation', u'Youth', u'Islamic Finance',
		u'Job Creation', u'Fair Trade', u'Kiva City LA',
		u'Innovative Loans', u'Growing Businesses', u'Disaster recovery',
		u'Health', u'Kiva City Detroit', 'Flexible Credit Study', 'none'])

		self.columns = []

	def get_themes(self, df):
		# filling null themes with emplty list so it doesn't throw errors
		df.themes = df.themes.map(lambda x: x if type(x) == list else ['none'])
		for theme in self.theme_list:
			#creating a dummy variable for each of the 22 themes
			df['theme_'+str(theme)] = df.themes.map(lambda x: theme in x)
		df = df.drop(['themes'],axis=1) 
		return df

	def transform(self, df):
		df = self.get_themes(df)
		sdum = pd.get_dummies(df.sector)
		cdum = pd.get_dummies(df.country)
		df = pd.concat([df,sdum,cdum],axis=1)
		df = df.drop(['posted_date','sector','tags','use','country','activity'],axis=1) 
		return df

	def fit_oversample(self, X,y,ratio=1,split =4):
		X_expired, y_expired = X[y], y[y]
		X_funded, y_funded = X[y==False], y[y==False]
		funded_prop = y_expired.shape[0]/float(y_funded.shape[0])
		X_sample, X_junk, y_sample, y_junk = train_test_split(X_funded,y_funded,train_size = funded_prop*ratio) 
		# print funded_prop
		# print X_sample.shape, y_sample.shape
		X_oversamp = np.concatenate([X_expired,X_sample],axis=0)
		y_oversamp = np.concatenate([y_expired,y_sample],axis=0)
		# print y_oversamp.mean()
		self.model = RandomForestClassifier(n_estimators=15,min_samples_split=split)
		self.model.fit(X_oversamp,y_oversamp)
		# print 'train score', rf.score(X_oversamp,y_oversamp)

	def fit_weighted(self, X, y,split=500, w = 2, leaf=20):
		print "doing weighted"
		self.model = RandomForestClassifier(min_samples_split=split, n_estimators=20, min_samples_leaf=leaf)
		weights = y/(y.mean()/w) + 1
		# self.model.fit(X, y, sample_weight = balance_weights(y)-.05)
		self.model.fit(X, y, sample_weight = weights)


	def transform_fit(self, df, mod = 'weighted', weight = 2, leaf=20):
		df = df.copy()
		print len(df.columns)
		y = df.pop('expired').values
		df = self.transform(df)
		X = df.values
		self.columns = df.columns
		if mod == 'weighted':
			self.fit_weighted(X,y,w=weight,leaf=leaf)
		else:
			self.fit_oversample(X,y)
		# print self.columns
		print len(self.columns)


	def predict (self, df):
		df = df.copy()
		y = df.pop('expired').values
		df = self.transform(df)
		df_cols = set(df.columns)
		print len(df_cols)
		fit_cols = set(self.columns)
		print len(fit_cols)
		new_cols = fit_cols.difference(df_cols)
		print len(new_cols)
		del_cols = df_cols.difference(fit_cols) 
		print len(del_cols)
		df = df.drop(list(del_cols),axis=1)
		print len(df.columns)
		for new_col in new_cols:
			df[new_col] = 0
		print len(df.columns)
		X = df.values
		ypred = self.model.predict(X)
		self.confusion(ypred, y)


	def confusion(self, ypred, y_test):
		# print ypred.mean()
		true_pos = np.logical_and(y_test,ypred).mean()
		false_pos = np.logical_and(y_test==False,ypred).mean()
		true_neg = np.logical_and(y_test==False,ypred==False).mean()
		false_neg = np.logical_and(y_test,ypred==False).mean()
		recall = true_pos / (true_pos + false_neg)
		precision = true_pos / (true_pos + false_pos)
		print 'pct_positive', y_test.mean() 
		print 'true_pos:', round(true_pos,4), 'false_pos:', round(false_pos,4)
		print 'true_neg', round(true_neg,4), 'false_neg', round(false_neg,4)
		print 'recall:', round(recall,4), 'precision:', round(precision,4)

	def feat_imp(self):
		# print self.columns
		column_list = self.columns
		# print len(column_list)
		imp = self.model.feature_importances_
		# print len(imp)
		most = np.argsort(imp)
		# print most
		most = most[::-1]
		# print column_list
		for feat in most:
			print str(column_list[feat]) + ": " +str(round(imp[feat] * 100,1)) + '%'


def run_model():
	df = pipe.load_cleaned(['timeframe'])
	mod = funding_model()
	mod.transform_fit(df)
	mod.feat_imp()

if __name__ == '__main__':
	run_model()