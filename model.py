import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


def timeframe(df):
	end = df[df.days_available == 45].posted_date.min()
	start = df[df.days_available == 60].posted_date.max()
	df = df[df.posted_date > start]
	df = df.drop(['days_available'],axis=1)
	return df[df.posted_date < end]

def binarize(df):
	df['dollars'] = df.currency_loss == 'none'
	df['currency_loss'] = df.currency_loss == 'partner'
	df['gender'] = df.gender == 'F'
	df['irregular_payments'] = df.repayment_interval == 'Irregularly'
	df['pay_at_end'] = df.repayment_interval == 'At end of term'
	df = df[df.status != 'refunded']
	df['expired'] = df.status == 'expired'
	df = df.drop(['repayment_interval','status',],axis=1)
	return df
	
def oversample_model(df):
	y = df.pop('expired').values
	X = df.values
	X_train, X_test, y_train, y_test = train_test_split(X,y)
	X_expired, y_expired = X_train[y_train], y_train[y_train]
	X_funded, y_funded = X_train[y_train==False], y_train[y_train==False]
	funded_prop = y_expired.shape[0]/float(y_funded.shape[0])
	X_sample, X_junk, y_sample, y_junk = train_test_split(X_funded,y_funded,train_size = funded_prop) 
	print X_test.shape
	print y_test.shape
	print X_funded.shape
	print y_funded.shape
	print X_expired.shape
	print y_expired.shape
	print funded_prop
	print X_sample.shape
	print y_sample.shape
	X_oversamp = np.concatenate([X_expired,X_sample],axis=0)
	y_oversamp = np.concatenate([y_expired,y_sample],axis=0)
	print y_oversamp.mean()
	rf = RandomForestClassifier()
	rf.fit(X_oversamp,y_oversamp)
	print 'train score', rf.score(X_oversamp,y_oversamp)
	print 'test score', rf.score(X_test, y_test)
	ypred = rf.predict(X_test)
	print ypred.mean()
	true_pos = np.logical_and(y_test,ypred).mean()
	false_pos = np.logical_and(y_test==False,ypred).mean()
	true_neg = np.logical_and(y_test==False,ypred==False).mean()
	false_neg = np.logical_and(y_test,ypred==False).mean()
	recall = true_pos / (true_pos + false_neg)
	precision = true_pos / (true_pos + false_pos)
	print 'pct_positive', y_test.mean() 
	print true_pos
	print false_pos
	print true_neg
	print false_neg
	print 'recall', recall
	print 'precision', precision

def dum(df):
	sdum = pd.get_dummies(df.sector)
	cdum = pd.get_dummies(df.country)
	df = pd.concat([df,sdum,cdum],axis=1)
	df = df.drop(['posted_date','sector','tags','themes','use','country','activity'],axis=1) 
	return df

def expired_plot(df):
	funded = (df[df.status!='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
	expired = (df[df.status=='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
	z = list(np.arange(2012.1,2015,1./12))
	plt.hist(funded,bins=z)
	plt.hist(expired,bins=z)
	plt.show()

def get_xy(df):
	y = df.pop('expired').values
	X = df.values
	return X,y


def expr_time(df):
	plt.scatter((df.posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014,df.days_available)
	plt.show()

def run_model():
	df = pipe.load_cleaned(['timeframe'])
	df = dum(df)
	oversample_model(df)
