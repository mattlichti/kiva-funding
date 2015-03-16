import pandas as pd
import matplotlib.pyplot as plt


def load_df():
	df = pd.io.pickle.read_pickle('transformed2.pkl')
	return df

def bla(df):
	print df[df['use: pay']>0].expired.mean()
	print df[df['use: pay']==0].expired.mean()
	print df[df['use: purchase']>0].expired.mean()
	print df[df['use: purchase']==0].expired.mean()
	print df[df.use_text_len > 190].expired.mean()
	print df[df.use_text_len < 190].expired.mean()
	print df[df['use: additional']>0].expired.mean()





