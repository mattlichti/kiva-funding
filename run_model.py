import data_pipeline
import build_model
import pandas as pd



def run(user, pw, db='kiva', table='loans', host='localhost', port='5432'):
	select = '''activity, bonus_credit_eligibility, loan_amount, sector,
		  themes, use, repayment_interval, repayment_term, currency_loss,
		  country, group_size, gender, expired'''
	where = "WHERE posted_date > (SELECT max(posted_date) FROM "+table \
			+ " WHERE days_available = 45) AND posted_date < '2015-02-01'" 
	dates = ['2015-02-01','2015-05-05']
	pipe = data_pipeline.Pipeline()
	pipe.setup_sql(user, pw, db=db, host=host, port=port, tables=table)
	pipe.load_from_sql(select=select, where=where)
	print pipe.df.info()
	mod = build_model.FundingModel()
	mod.transform_fit(pipe.df)
	pipe.load_from_sql(select=select, date_range = dates)
	print pipe.df.info()
	y = pipe.df.pop('expired').values
	ypred = mod.predict(pipe.df)
	mod.confusion(ypred, y)
	mod.feat_imp()