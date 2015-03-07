import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pipe

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

def expired_plot(df):
	funded = (df[df.status!='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
	expired = (df[df.status=='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
	z = list(np.arange(2012.1,2015,1./12))
	plt.hist(funded,bins=z)
	plt.hist(expired,bins=z)
	plt.show()

def expr_time(df):
	plt.scatter((df.posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014,df.days_available)
	plt.show()

def timeframe2014():
	df = pipe.load_cleaned(['everything'])
	df = timeframe(df)
	df = binarize(df)
	pipe.dump(df,'timeframe.json')
