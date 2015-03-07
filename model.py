import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

theme_list = sorted([u'Underfunded Areas', u'Rural Exclusion',
       u'Vulnerable Groups', u'Conflict Zones', u'Mobile Technology',
       u'Green', u'Higher Education', u'Start-Up', u'Arab Youth', u'SME',
       u'Water and Sanitation', u'Youth', u'Islamic Finance',
       u'Job Creation', u'Fair Trade', u'Kiva City LA',
       u'Innovative Loans', u'Growing Businesses', u'Disaster recovery',
       u'Health', u'Kiva City Detroit', 'Flexible Credit Study'])


def oversample_model(X,y,ratio=1):
	X_train, X_test, y_train, y_test = train_test_split(X,y)
	X_expired, y_expired = X_train[y_train], y_train[y_train]
	X_funded, y_funded = X_train[y_train==False], y_train[y_train==False]
	funded_prop = y_expired.shape[0]/float(y_funded.shape[0])
	X_sample, X_junk, y_sample, y_junk = train_test_split(X_funded,y_funded,train_size = funded_prop*ratio) 
	print X_test.shape, y_test.shape
	print X_funded.shape, y_funded.shape
	# print X_expired.shape, y_expired.shape
	# print funded_prop
	# print X_sample.shape, y_sample.shape
	X_oversamp = np.concatenate([X_expired,X_sample],axis=0)
	y_oversamp = np.concatenate([y_expired,y_sample],axis=0)
	# print y_oversamp.mean()
	rf = RandomForestClassifier()
	rf.fit(X_oversamp,y_oversamp)
	# print 'train score', rf.score(X_oversamp,y_oversamp)
	# print 'test score', rf.score(X_test, y_test)
	ypred = rf.predict(X_test)
	confusion(ypred, y_test)
	return rf

def feat_imp(fit_model, column_list):
	imp = fit_model.feature_importances_
	print imp
	most = np.argsort(imp)
	print most
	most = most[::-1]
	print column_list
	for feat in most:
		print str(column_list[feat]) + ": " +str(round(imp[feat] * 100,1)) + '%'

def confusion(ypred, y_test):
	# print ypred.mean()
	true_pos = np.logical_and(y_test,ypred).mean()
	false_pos = np.logical_and(y_test==False,ypred).mean()
	true_neg = np.logical_and(y_test==False,ypred==False).mean()
	false_neg = np.logical_and(y_test,ypred==False).mean()
	recall = true_pos / (true_pos + false_neg)
	precision = true_pos / (true_pos + false_pos)
	print 'pct_positive', y_test.mean() 
	print 'true_pos:', round(true_pos,4), 'false_pos:', round(false_pos,4)
	print 'true_neg', true_neg, 'false_neg', false_neg
	print 'recall:', recall, 'precision:', precision

def get_themes(df):
	# filling null themes with emplty list so it doesn't throw errors
	df.themes = df.themes.map(lambda x: x if type(x) == list else [])
	for theme in theme_list:
		#creating a dummy variable for each of the 22 themes
		df[theme] = df.themes.map(lambda x: theme in x)
	df = df.drop(['themes'],axis=1) 
	return df

def dum(df):
	df = get_themes(df)
	sdum = pd.get_dummies(df.sector)
	cdum = pd.get_dummies(df.country)
	df = pd.concat([df,sdum,cdum],axis=1)
	df = df.drop(['posted_date','sector','tags','use','country','activity'],axis=1) 
	return df

def get_xy(df):
	y = df.pop('expired').values
	X = df.values
	return X,y

def run_model():
	df = pipe.load_cleaned(['timeframe'])
	df = dum(df)
	x,y = get_xy(df)
	column_list = df.columns
	rf = oversample_model(x,y)
	feat_imp(rf, column_list)

if __name__ == '__main__':
	run_model()