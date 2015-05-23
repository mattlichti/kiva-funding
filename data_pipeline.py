import pandas as pd
import numpy as np
import json
import glob
from sqlalchemy import create_engine
import psycopg2
from collections import defaultdict

class Pipeline(object):

    def __init__(self):
        self.drop_cols = ['basket_amount', 'currency_exchange_loss_amount',
                         'delinquent', 'paid_date', 'paid_amount', 'payments',
                         'lender_count', 'funded_amount',
                         'translator', 'video', 'tags', 'journal_totals']

        self.themes = ('Underfunded Areas', 'Rural Exclusion', 'Start-Up',
                       'Vulnerable Groups', 'Conflict Zones', 'Youth', 'SME',
                       'Mobile Technology', 'Green', 'Job Creation', 'Health',
                       'Higher Education', 'Arab Youth', 'Islamic Finance', 
                       'Water and Sanitation', 'Fair Trade', 'Kiva City LA',
                       'Innovative Loans', 'Growing Businesses',
                       'Disaster recovery', 'Kiva City Detroit',
                       'Flexible Credit Study', 'none')

        self.min_date = '2012-01-25'  # date kiva expiration policy implemented
        self.df = pd.DataFrame()  # pandas dataframe of loan info
        self.sql = defaultdict()  # info for connecting to postgres db
        self.tables = []  # sql tables storing the loan info
        self.sql_engine = None  # used with sqlalchemy

    def import_loans(self, files=None, folder=None):
        '''
        Input either a list of json loan files from the kiva api or the address
        of a folder containing said files. Creates pandas df of the kiva loans
        '''
        self.df = pd.DataFrame()
        if folder:
            files = glob.glob(folder + "/*.json")
        for file in files:
            f = open(file)
            dic = json.loads(f.read())
            df = pd.DataFrame(dic['loans'])
            df['date_fetched'] = dic['header']['date']
            self.df = pd.concat([self.df, df], axis=0, ignore_index=True)

    def get_desc(self):
        '''
        extracts the English description and drops the description
        in other languages
        '''
        self.df['description'] = self.df.description.map(lambda x: \
                x['texts']['en'] if 'en' in x['languages'] else '')
        self.df['use_text_len'] = self.df.use.map(lambda x: len(x))
        self.df['desc_text_len'] = self.df.description.map(lambda x: len(x))

    def transform_themes(self):
        self.df.themes = self.df.themes.map(lambda x: x if type(x) == list \
                         else ['none'])
        for theme in self.themes:
            self.df['theme_'+str(theme).replace(' ','_').lower()] = \
                self.df.themes.map(lambda x: theme in x)
        self.df.drop('themes', axis=1, inplace=True)

    def payment_terms(self):
        '''
        Extracts repayment interval (monthly, irregularly, or lump sum),
        repayment term (in months), and potential currency loss info
        and drops the rest of the repayment info
        '''
        self.df['repayment_interval'] = self.df.terms.map(lambda x: \
                                        x['repayment_interval'])
        self.df['repayment_term'] = self.df.terms.map(lambda x: \
                                    x['repayment_term'])
        self.df['currency_loss'] = self.df.terms.map(lambda x: \
                    x['loss_liability']['currency_exchange']=='shared')
        self.df.drop('terms', axis=1, inplace=True)

    def borrower_info(self):
        '''
        Extracts country, group size, gender, and drops other borrower info
        '''
        self.df['country'] = self.df.location.map(lambda x: x['country_code'])
        self.df['group_size'] = self.df.borrowers.map(lambda x: len(x))
        self.df['image'] = self.df.image.map(lambda x: x['id'])
        self.df['gender'] = self.df.borrowers.map(lambda x: x[0]['gender']=='F')
        self.df['anonymous'] = self.df.name.map(lambda x: x == 'Anonymous')
        self.df.drop(['borrowers', 'location'], axis=1, inplace=True)

    def transform_dates(self):
        '''
        Converts posted date to datetime object
        calculates the number of days the loan is available on kiva from the
        posted date until planned expiration date as timedelta object
        '''
        self.df['posted_date'] = pd.to_datetime(self.df.posted_date)
        self.df['date_fetched'] = pd.to_datetime(self.df.date_fetched)
        self.df['planned_expiration_date'] = pd.to_datetime(
            self.df.planned_expiration_date)
        self.df['days_available'] = ((self.df.planned_expiration_date -
                                     self.df.posted_date)/np.timedelta64
                                     (1, 'D')).round().astype(int)
        self.df['end_date'] = pd.to_datetime(self.df.funded_date)
        self.df['end_date'][self.df.end_date.isnull()] = self.df.planned_expiration_date
        self.df.drop('funded_date', axis=1, inplace=True)

    def filter_by_date(self):
        '''
        filters out loans with a planned expiration date after the data
        was fetched and loans posted before kiva expiration policy implemented
        '''
        self.df = self.df[(self.df.planned_expiration_date < self.df.date_fetched)
                          & (self.df.posted_date > self.min_date)]

    def transform_labels(self):
        '''
        labels loans as either expired or funded. Drops the current status
        info and drops the tiny number of loans which were refunded
        (withdrawn from kiva without being funded or expiring)
        '''
        self.df['expired'] = self.df.status == 'expired'
        self.df = self.df[self.df.status != 'refunded']
        self.df.drop('status', axis=1, inplace=True)

    def transform_df(self):
        '''
        Loads and transforms the data, drops data that was generated after
        loan was posted
        '''
        self.df = self.df.drop_duplicates(['id'])
        self.df = self.df.set_index('id')
        self.df.drop(self.drop_cols, axis=1, inplace=True)
        self.transform_dates()
        self.filter_by_date()
        self.get_desc()
        self.payment_terms()
        self.borrower_info()
        self.transform_themes()
        self.transform_labels()

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
        engstr = 'postgresql://%s:%s@%s:%s/%s' %(user, pw, host, port, db)
        self.sql_engine = create_engine(engstr)

    def export_to_sql(self, table):
        '''
        input name of table to insert the pandas dataframe self.df
        using the sql db, user, and pw input by the setup sql function
        '''
        self.df.to_sql(table, self.sql_engine)
        self.tables.append(table)

    def load_from_sql(self, table=None, where='', date_range=None, columns='*'):
        if not table:
            table = self.tables[-1]
        if date_range:
            where = '''WHERE posted_date > '%s' 
                       AND posted_date <  '%s' ''' % (date_range[0], date_range[1])
        query = 'SELECT %s FROM %s %s;' % (columns, table, where)
        self.df = pd.read_sql_query(query, self.sql_engine)

    def merge_db(self, new_tab):
        '''
        merge multiple sql dbs into 1 db. Used by sql_pipeline function to
        transform and export loans to sql in batches
        '''
        conn = psycopg2.connect(dbname=self.sql['db'], user=self.sql['user'],
                                host=self.sql['host'], password=self.sql['pw'])
        c = conn.cursor()
        query = '''DROP TABLE IF EXISTS %s; 
                    CREATE VIEW merged AS ''' % new_tab
        for tab in self.tables[:-1]:
            query +=    '(SELECT * FROM %s) UNION ' % tab
        query +=        '(SELECT * FROM %s); ' % self.tables[-1]
        query += '''CREATE VIEW supply as 
                        SELECT count(1) loans_on_site, merged.id 
                        FROM merged JOIN (
                            SELECT posted_date, end_date FROM merged) a
                        ON merged.posted_date > a.posted_date 
                        AND merged.posted_date < a.end_date 
                        GROUP BY merged.id; 
                    CREATE TABLE %s as 
                        SELECT * FROM supply join merged using (id);  
                    DROP VIEW supply; 
                    DROP VIEW merged;''' % new_tab
        for tab in self.tables:
            query += ' DROP TABLE %s; ' % tab
        c.execute(query)
        conn.commit()
        conn.close()
        self.tables = [new_tab]

    def sql_pipeline(self, address, user, pw, table_name='loans', batch=40):
        '''
        imports, transforms, and loads loan data into sql
        address is folder containing json files from kiva api
        batch is the number of json files to transform at a time
        default 40 files = 20,000 kiva loan records which is about 120 MB
        '''
        filelist = glob.glob(address + "/*.json")
        self.setup_sql(user, pw)
        for n in xrange(0, (len(filelist)-1)/batch+1):
            self.import_loans(files=filelist[n*batch:(n+1)*batch])
            self.transform_df()
            self.export_to_sql('temp_' + str(n))
        self.merge_db(table_name)
