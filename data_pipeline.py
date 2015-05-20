import pandas as pd
import numpy as np
import json
import glob
from sqlalchemy import create_engine
import psycopg2


class Pipeline(object):

    def __init__(self):
        # drop data generated after loan was posted - can't be used in model
        self.droplist = ['basket_amount', 'currency_exchange_loss_amount',
                         'delinquent', 'paid_date', 'paid_amount', 'payments',
                         'lender_count', 'funded_date', 'funded_amount',
                         'translator', 'video', 'tags', 'journal_totals']

        self.min_date = '2012-01-25'  # date kiva expiration policy implemented
        self.date_fetched = None  # date the loan info fetched from kiva API
        self.df = None  # pandas dataframe of loan info
        self.sql = {'user': None, 'pw': None, 'db': None, 'host': None,
                    'port': None}
        self.tables = []  # sql tables storing the loan info
        self.sql_engine = None  # used with sqlalchemy

    def import_loans(self, files=None, folder=None):
        '''
        Input either a list of json loan files from the kiva api or the address
        of a folder containing said files. Creates pandas df of the kiva loans
        '''
        if folder:
            files = glob.glob(folder + "/*.json")
        lst = []
        for file in files:
            f = open(file)
            dic = json.loads(f.readline())
            lst += (dic['loans'])
        self.date_fetched = str(dic['header']['date'])[0:10]
        self.df = pd.DataFrame(lst)

    def get_desc(self):
        '''
        extracts the English description and drops the description
        in other languages
        '''
        text_df = pd.DataFrame(list(self.df.description.map(lambda x:
                                                            x['texts'])))
        self.df['description'] = text_df['en']  # null for text without English

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
        self.df.drop('terms', axis=1, inplace=True)

    def borrower_info(self):
        '''
        Extracts country, group size, gender, and drops other borrower info
        '''
        self.df['country'] = self.df.location.map(lambda x: x['country_code'])
        self.df['group_size'] = self.df.borrowers.map(lambda x: len(x))
        self.df['gender'] = self.df.borrowers.map(lambda x: x[0]['gender'])
        self.df.drop(['borrowers', 'location'], axis=1, inplace=True)

    def transform_dates(self):
        '''
        Converts posted date to datetime object
        calculates the number of days the loan is available on kiva from the
        posted date until planned expiration date as timedelta object
        '''
        self.df['posted_date'] = pd.to_datetime(self.df.posted_date)
        self.df['planned_expiration_date'] = pd.to_datetime(
            self.df.planned_expiration_date)
        self.df['days_available'] = ((self.df.planned_expiration_date -
                                     self.df.posted_date)/np.timedelta64
                                     (1, 'D')).round().astype(int)

    def filter_by_date(self):
        '''
        filters out loans with a planned expiration date after the data
        was fetched and loans posted before kiva expiration policy implemented
        '''
        self.df = self.df[(self.df.planned_expiration_date < self.date_fetched)
                          & (self.df.posted_date > self.min_date)]
        self.df.drop('planned_expiration_date', axis=1, inplace=True)

    def transform_labels(self):
        '''
        labels loans as either expired or funded. Drops the current status
        info and drops the tiny number of loans which were refunded
        (withdrawn from kiva without being funded or expiring)
        '''
        self.df['expired'] = self.df.status == 'expired'
        self.df = self.df[self.df.status != 'refunded']
        self.df = self.df.drop('status', axis=1)

    def transform_df(self):
        '''
        Loads and transforms the data, drops data that was generated after
        loan was posted
        '''
        self.df = self.df.drop_duplicates(['id'])
        self.df.drop(self.droplist, axis=1, inplace=True)
        self.get_desc()
        self.payment_terms()
        self.borrower_info()
        self.transform_dates()
        self.filter_by_date()
        self.transform_labels()
        self.df['image'] = self.df.image.map(lambda x: x['id'])
        self.df = self.df.set_index('id')

    def setup_sql(self, user, pw, db='kiva', host='localhost', port='5432',
                  tables=[]):
        '''
        sets up sql connection for exporting and loading from sql
        must be run before export_to_sql, load_from_sql, or merge_db
        '''
        self.sql['db'] = db
        self.sql['user'] = user
        self.sql['pw'] = pw
        self.sql['host'] = host
        self.sql['port'] = str(port)
        self.tables = tables
        engstr = 'postgresql://' + user+':'+pw+'@'+host + ':' + port + '/' + db
        self.sql_engine = create_engine(engstr)

    def export_to_sql(self, table):
        '''
        input name of table to insert the pandas dataframe self.df
        using the sql db, user, and pw input by the setup sql function
        '''
        self.df.to_sql(table, self.sql_engine)
        self.tables.append(table)

    def load_from_sql(self, table=None, where='', date_range=None, select='*'):
        if table:
            self.tables = table
        if date_range:
            where = "WHERE posted_date > '" + date_range[0] \
                    + "' AND posted_date < '" + date_range[1] + "'"
        query = 'SELECT '+select+' FROM '+self.tables+' '+where+' ;'
        self.df = pd.read_sql_query(query, self.sql_engine)

    def merge_db(self, new_tab):
        '''
        merge multiple sql dbs into 1 db. Used by sql_pipeline function to
        transform and export loans to sql in batches
        '''
        conn = psycopg2.connect(dbname=self.sql['db'], user=self.sql['user'],
                                host=self.sql['host'], password=self.sql['pw'])
        c = conn.cursor()
        query = 'DROP TABLE IF EXISTS ' + new_tab + '; '
        query += 'Create TABLE ' + new_tab + ' AS '
        for tab in self.tables[:-1]:
            query += '(SELECT * FROM ' + tab + ') union '
        query += '(SELECT * FROM ' + self.tables[-1] + ');'
        for tab in self.tables:
            query += ' DROP TABLE ' + tab + ';'
        c.execute(query)
        conn.commit()
        conn.close()
        self.tables = new_tab

    def sql_pipeline(self, address, user, pw, table_name='loans', batch=50):
        '''
        imports, transforms, and loads loan data into sql
        address is folder containing json files from kiva api
        batch is the number of json files to transform at a time
        default 50 files = 25,000 kiva loan records at 150 MB
        '''
        filelist = glob.glob(address + "/*.json")
        self.setup_sql(user, pw)
        for n in xrange(0, (len(filelist)-1)/batch+1):
            self.import_loans(files=filelist[n*batch:(n+1)*batch])
            self.transform_df()
            self.export_to_sql('temp_' + str(n))
        self.merge_db(table_name)
