import data_pipeline
import build_model
import pandas as pd
import sys


def run(user, pw, db='kiva', table='loans', host='localhost', port='5432',
        train_dates=('2014-01-01', '2014-12-01'),
        test_dates=('2015-01-01', '2015-05-01')):
    '''
    Loads kiva loan data from sql table, trains model on loans within
    train_dates, tests_model on loans from within test_dates,
    Prints the most important features.
    '''
    pipe = data_pipeline.Pipeline()
    cols = '''activity, bonus_credit_eligibility, loan_amount, sector,
              use, repayment_interval, repayment_term, currency_loss,
              country, group_size, gender, desc_text_len, use_text_len,
              anonymous, competing_loans, days_available, expired, '''
    cols += ", ".join('"theme_'+theme.replace(' ', '_').lower()+'"'
                      for theme in pipe.themes)

    pipe.setup_sql(user, pw, db=db, host=host, port=port)
    pipe.load_from_sql(cols=cols, date_range=train_dates, table=table)
    mod = build_model.FundingModel()
    mod.fit(pipe.df)

    pipe.load_from_sql(cols=cols, date_range=test_dates, table=table)
    y = pipe.df.pop('expired').values
    ypred = mod.predict(pipe.df)
    print mod.confusion_matrix(ypred, y)
    print '\n Feature Importance \n'
    print mod.feat_imp()

if __name__ == '__main__':
    '''
    Input postgres username and password. Optional: input postgres
    db name, host, port, batch size, competing_loans boolean, table name
    ex: python run_model.py username password
    '''
    run(*sys.argv[1:])
