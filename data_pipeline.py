import pandas as pd
import numpy as np
import json
import glob
import os
from sqlalchemy import create_engine
import psycopg2


class Pipeline(object):

    def __init__(self):
        self.droplist = ['basket_amount','currency_exchange_loss_amount',
                        'delinquent', 'paid_date', 'paid_amount','payments',
                        'lender_count', 'funded_date','funded_amount',
                        'translator','video', 'tags', 'journal_totals']
        '''droplist mostly contains data that can't be used in model because
        it was generated after loan was posted'''
        self.min_date = '2012-01-25' # date kiva expiration policy implemented
        self.date_fetched = None # date the loan info fetched from kiva API
        self.df = None # pandas dataframe of loan info

    def import_loans(self, address):
        '''
        Input address of folder containing raw json files from kiva api
        Creates pandas dataframe of the kiva loans
        ''' 
        lst = []
        for file in glob.glob(address + "/*.json"):
            f = open(file)
            dic = json.loads(f.readline())
            lst +=(dic['loans'])
        self.date_fetched = str(dic['header']['date'])[0:10]
        self.df = pd.DataFrame(lst)
        self.df = self.df.drop_duplicates(['id'])   

    def get_desc(self):
        '''
        extracts the English description and drops the description
        in other languages
        '''
        text_df = pd.DataFrame(list(self.df.description.map(lambda x: x['texts'])))
        self.df['description'] = text_df['en'] # null for text without any English

    def payment_terms(self):
        '''
        Extracts repayment interval (monthly, irregularly, or lump sum),
        repayment term (in months), and potential currency loss info 
        and drops the rest of the repayment info
        '''
        terms = pd.DataFrame(list(self.df.terms))
        self.df['repayment_interval'] = terms['repayment_interval']
        self.df['repayment_term'] = terms['repayment_term']
        curr_loss = pd.DataFrame(list(terms.loss_liability))
        self.df['currency_loss'] = curr_loss.currency_exchange
        self.df.drop('terms',axis=1,inplace=True)

    def borrower_info(self):
        ''' 
        Extracts country, group size, gender, and drops other borrower info
        '''
        self.df['country'] = self.df.location.map(lambda x: x['country_code'])
        self.df['group_size'] = self.df.borrowers.map(lambda x: len(x))
        self.df['gender'] = self.df.borrowers.map(lambda x: x[0]['gender'])
        self.df.drop(['borrowers', 'location'],axis=1,inplace=True)

    def transform_dates(self):
        '''
        Converts posted date to datetime object
        calculates the number of days the loan is available on kiva from the 
        posted date until planned expiration date as timedelta object
        '''
        self.df['posted_date'] = pd.to_datetime(self.df.posted_date)
        self.df['planned_expiration_date'] = pd.to_datetime(self.df.planned_expiration_date)
        self.df['days_available'] = ((self.df.planned_expiration_date - self.df.posted_date)/
                                np.timedelta64(1, 'D')).round().astype(int)

    def filter_by_date(self):
        ''' 
        filters out loans with a planned expiration date after the data 
        was fetched and loans posted before kiva expiration policy implemented
        '''
        self.df = self.df[(self.df.planned_expiration_date < 
        self.date_fetched) & (self.df.posted_date > self.min_date)]
        self.df.drop('planned_expiration_date',axis=1,inplace=True)

    def build_df(self, simple_mod = True):
        '''
        Loads and transforms the data, drops data that was generated after 
        loan was posted 
        '''
        self.df = self.df.drop_duplicates(['id'])      
        self.df.drop(self.droplist,axis=1,inplace=True)
        self.payment_terms()
        self.borrower_info()
        self.transform_dates()
        self.filter_by_date()

        if simple_mod:
            self.df.drop(['image', 'name', 'partner_id', 'description'], axis=1, inplace=True)
        else: # model that uses description text
            self.get_desc()
            self.df['image'] = self.df.image.map(lambda x: x['id'])
        self.df = self.df.set_index('id')

    def export_to_sql(self, table, password):
        engine_string = 'postgresql://matt:' + password + '@localhost:5432/kiva'
        engine = create_engine(engine_string)
        self.df.to_sql(table, engine)

    def load_from_sql(self, table, password):
        engine_string = 'postgresql://matt:' + password + '@localhost:5432/kiva'
        engine = create_engine(engine_string)
        query = 'SELECT * FROM ' + table + ';'
        self.df = pd.read_sql_query(query, engine)

    def export_to_pickle(self, filename):
        ''' save cleaned dataframe as pickle'''
        self.df.to_pickle('data/pickles/' + filename + '.pkl')

    def load_from_pickles(self, lst):
        '''input list of filenames of pickled dataframes
           creates dataframe that merges all the pickled dfs'''
        dfs = []
        for pick in lst:
            df = pd.io.pickle.read_pickle('data/pickles/'+ pick +'.pkl')
            dfs.append(df)
        self.df = pd.concat(dfs,axis=0)
        self.df['id'] = self.df.index
        self.df = self.df.drop_duplicates(['id'])        
        self.df = self.df.set_index('id')

    def export_to_json(self, filename):
        '''save cleaned dataframe as json'''
        loans = self.df.to_json()
        with open(filename, 'w') as outfile:
            json.dump(loans, outfile)

    def mergedb(self, pw, tablist, new_tab):
        conn = psycopg2.connect(dbname='kiva', user='matt', host='localhost', password = pw)
        c = conn.cursor()
        query = 'DROP TABLE IF EXISTS ' + new_tab + '; '
        query+= 'Create TABLE ' + new_tab + ' AS '
        for tab in tablist[:-1]:
            query +=  '(SELECT * FROM ' + tab + ') union '
        query +=  '(SELECT * FROM ' + tablist[-1] + ');'
        for tab in tablist:
            query += ' DROP TABLE ' + tab + ';' 
        c.execute(query)
        conn.commit()
        conn.close()

    def load_and_export_to_sql(self, address, pw, batch = 50):
        '''
        imports, transforms, and loads loan data into sql
        ''' 
        lst = []
        dblst = []
        for n, file in enumerate(glob.glob(address + "/*.json")):
            f = open(file)
            dic = json.loads(f.readline())
            lst +=(dic['loans'])
            print n
            if (n%batch == (batch-1)) | (n == (len(glob.glob(address + "/*.json"))-1)):
                self.date_fetched = str(dic['header']['date'])[0:10]
                self.df = pd.DataFrame(lst)
                self.build_df()
                self.export_to_sql('temp_' + str(n), pw)
                dblst.append('temp_' + str(n))
                lst = []
                print dblst
        self.mergedb(pw,dblst,'mmmm')


def run_sql(pw, lst = ['1601','1602','1100s', '1200s']):
    p = Pipeline()
    for folder in lst:
        p.import_loans('data/loans/' + folder)
        p.build_df()
        print 'built df'
        p.export_to_sql('temp_' + folder, pw)
        print 'in sql'
    p.mergedb(pw,['temp_' + item for item in lst],'mdb')
    