import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data_pipeline

def load_transformed():
    df = pd.io.pickle.read_pickle('data/transformed2.pkl')
    return df

def load_untrans():
    df = pd.io.pickle.read_pickle('data/2014_df.pkl')
    return df



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

def timeframe2014():
	df = data_pipeline.load_cleaned(['everything'])
	df = timeframe(df)
	df = binarize(df)
	data_pipeline.dump(df,'timeframe.json')

def pickl2014():
	df = data_pipeline.load_cleaned(['everything'])
	df = timeframe(df)
	df['expired'] = df.status == 'expired'
	df = df[df.status != 'refunded']
	df = df.drop(['status','tags'],axis=1)
	df.to_pickle('2014.pkl')

def get_months(df):
	jan = df[(df.posted_date > '2014-1-01') & (df.posted_date < '2014-2-01')]
	feb = df[(df.posted_date > '2014-2-01') & (df.posted_date < '2014-3-01')]
	march = df[(df.posted_date > '2014-3-01') & (df.posted_date < '2014-4-01')]
	april = df[(df.posted_date > '2014-4-01') & (df.posted_date < '2014-5-01')]
	may = df[(df.posted_date > '2014-5-01') & (df.posted_date < '2014-6-01')]
	june = df[(df.posted_date > '2014-6-01') & (df.posted_date < '2014-7-01')]
	july = df[(df.posted_date > '2014-7-01') & (df.posted_date < '2014-8-01')]
	aug = df[(df.posted_date > '2014-8-01') & (df.posted_date < '2014-9-01')]
	sep = df[(df.posted_date > '2014-9-01') & (df.posted_date < '2014-10-01')]
	octb = df[(df.posted_date > '2014-10-01') & (df.posted_date < '2014-11-01')]
	nov = df[(df.posted_date > '2014-11-01') & (df.posted_date < '2014-12-01')]
	return [jan, feb, march, april, may, june, july, aug, sep, octb, nov]

def get_time_periods(df):	
	augsep = df[(df.posted_date > '2014-8-01') & (df.posted_date < '2014-10-01')]
	apsep = df[(df.posted_date > '2014-4-01') & (df.posted_date < '2014-10-01')]
	augnov = df[(df.posted_date > '2014-8-01') & (df.posted_date < '2014-12-01')]
	apnov = df[df.posted_date > '2014-4-01']
	return apsep, augsep

def train_set(df):	
	'''trained on April through September loans. All loans in this timeframe
	   have 30 day expiration policy so all were funded or expired by Oct 30'''
	return df[(df.posted_date > '2014-4-01') & (df.posted_date < '2014-10-01')]

def test_set(df):
	'''testing on November up until expiration policy switched from normal
	   30 days to the Christmas 45 day expiration period'''
	return df[df.posted_date > '2014-11-01']


if __name__ == '__main__':
	pickl2014()