import pandas as pd
import numpy as np
import json
import glob
import os
from sqlalchemy import create_engine

class Pipeline(object):

    def __init__(self):
        self.droplist = ['basket_amount','currency_exchange_loss_amount','delinquent',
                        'paid_date', 'paid_amount', 'journal_totals', 'payments',
                        'lender_count', 'funded_date','funded_amount','translator', 
                        'video', 'tags']
        self.min_date = '2012-01-25' # date kiva expiration policy implemented


    def import_loans(self, address):
        '''
        Input address of folder full of json files
        Output dataframe of kiva loans
        ''' 
        lst = []
        for file in glob.glob(address + "/*.json"):
            f = open(file)
            dic = json.loads(f.readline())
            lst +=(dic['loans'])
        df = pd.DataFrame(lst)
        df = df.drop_duplicates(['id'])        
        return df

    def get_desc(self, df):
        '''
        extracts the English description and drops the description
        in other languages
        '''
        text_df = pd.DataFrame(list(df.description.map(lambda x: x['texts'])))
        df['description'] = text_df['en'] # null for text without any English
        return df

    def payment_terms(self, df):
        '''
        Extracts repayment interval (monthly, irregularly, or lump sum),
        repayment term (in months), and potential currency loss info 
        and drops the rest of the repayment info
        '''
        terms = pd.DataFrame(list(df.terms))
        df['repayment_interval'] = terms['repayment_interval']
        df['repayment_term'] = terms['repayment_term']
        curr_loss = pd.DataFrame(list(terms.loss_liability))
        df['currency_loss'] = curr_loss.currency_exchange
        df.drop('terms',axis=1,inplace=True)
        return df

    def borrower_info(self, df):
        ''' 
        Extracts country, group size, gender, and drops other borrower info
        '''
        df['country'] = df.location.map(lambda x: x['country_code'])
        df['group_size'] = df.borrowers.map(lambda x: len(x))
        df['gender'] = df.borrowers.map(lambda x: x[0]['gender'])
        df.drop(['borrowers', 'location'],axis=1,inplace=True)
        return df

    def transform_dates(self, df):
        '''
        Converts posted date to datetime object
        calculates the number of days the loan is available on kiva from the 
        posted date until planned expiration date as timedelta object
        '''
        df['posted_date'] = pd.to_datetime(df.posted_date)
        df['planned_expiration_date'] = pd.to_datetime(df.planned_expiration_date)
        df['days_available'] = ((df.planned_expiration_date - df.posted_date)/
                                np.timedelta64(1, 'D')).round().astype(int)
        df.drop('planned_expiration_date',axis=1,inplace=True)
        return df

    def build_df(self, address, model='simple'):
        '''
        Loads and transforms the data, drops the data that was generated after loan
        was funded (since that data will not be available for future predictions)
        '''
        df = self.import_loans(address)
        df.drop(self.droplist,axis=1,inplace=True)
        df = self.payment_terms(df)
        df = self.borrower_info(df)
        df = self.transform_dates(df)

        if model == 'complex': # model that uses description text
            df = self.get_desc(df)
            df['image'] = df.image.map(lambda x: x['id'])
        else:
            df.drop(['image', 'name', 'partner_id', 'description'], axis=1, inplace=True)
        
        df = df.set_index('id')
        return df

def filter_by_date(df):
    min_date = '2012-01-25' # when kiva expiration policy fully implemented
    max_date = '2014-12-22' # last day when all loans in dataset could expire
    df = df[(df.posted_date < max_date) & (df.posted_date > min_date)]
    return df

def dump_to_pickle(df, filename):
    ''' save cleaned dataframe as pickle'''
    df.to_pickle('data/' + filename + '.pkl')

def load_pickles(lst):
    '''input list of locations of pickled dataframes
       output dataframe that merges all the pickles
       useful when cleaning data in batches'''
    dfs = []
    for pick in lst:
        df = pd.io.pickle.read_pickle('data/pickles/'+ pick +'.pkl')
        dfs.append(df)
    df = pd.concat(dfs,axis=0)
    return df

def export_as_json(df, filename):
    '''save cleaned dataframe as json'''
    loans = df.to_json()
    os.chdir(address + 'dumps')
    with open(filename, 'w') as outfile:
        json.dump(loans, outfile)

def export_to_sql(df, table, password):
    engine_string = 'postgresql://matt:' + password + '@localhost:5432/kiva'
    engine = create_engine(engine_string)
    df.to_sql(table, engine)

def import_from_sql(password):
    engine_string = 'postgresql://matt:' + password + '@localhost:5432/kiva'
    engine = create_engine(engine_string)
    df = pd.read_sql_query('SELECT * FROM loans;', engine)
    return df
