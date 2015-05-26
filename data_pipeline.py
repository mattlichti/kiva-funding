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
                          'lender_count', 'funded_amount', 'translator',
                          'video', 'tags', 'journal_totals', 'terms',
                          'funded_date', 'borrowers', 'location', 'themes']

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
        self.query = ''

    def import_loans(self, files=None, folder=None):
        '''
        Input either a list of json loan files from the kiva api or the address
        of a folder containing said files. Creates pandas df of the kiva loans
        '''
        self.df = pd.DataFrame()
        if folder:
            files = glob.glob(folder + "/*.json")
        print len(files)
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
        self.df['description'] = self.df.description.map(
            lambda x: x['texts']['en'] if 'en' in x['languages'] else '')
        self.df['use_text_len'] = self.df.use.map(lambda x: len(x))
        self.df['desc_text_len'] = self.df.description.map(lambda x: len(x))

    def transform_themes(self):
        '''
        Themes are loan attributes that lenders can use to search for loans
        Loans can have one or more themes.
        '''
        self.df.themes = self.df.themes.map(
            lambda x: x if type(x) == list else ['none'])
        for theme in self.themes:
            self.df['theme_'+str(theme).replace(' ', '_').lower()] = \
                self.df.themes.map(lambda x: theme in x)

    def payment_terms(self):
        '''
        Extracts repayment interval (monthly, irregularly, or lump sum),
        repayment term (in months), and potential currency loss info
        and drops the rest of the repayment info
        '''
        self.df['repayment_interval'] = self.df.terms.map(
            lambda x: x['repayment_interval'])
        self.df['repayment_term'] = self.df.terms.map(
            lambda x: x['repayment_term'])
        self.df['currency_loss'] = self.df.terms.map(
            lambda x: x['loss_liability']['currency_exchange'] == 'shared')

    def borrower_info(self):
        '''
        Extracts country, group size, gender, image id, and whether the
        borrower is listed as anonymous on the site
        '''
        self.df['country'] = self.df.location.map(lambda x: x['country_code'])
        self.df['group_size'] = self.df.borrowers.map(lambda x: len(x))
        self.df['image'] = self.df.image.map(lambda x: x['id'])
        self.df['gender'] = self.df.borrowers.map(
            lambda x: x[0]['gender'] == 'F')
        self.df['anonymous'] = self.df.name.map(lambda x: x == 'Anonymous')

    def transform_dates(self):
        '''
        Converts dates to datetime objects.
        Drops loans posted before expiration policy implemented '2012-01-25'.
        Drops loans with planned expiration date after loan fetched from api.
        Calculates the number of days loan is available until expiration.
        Calculates the date when funding ended (expired or fully funded).
        Calculates number of days on site before loan was funded or expired.
        '''
        self.df['posted_date'] = pd.to_datetime(self.df.posted_date)
        self.df['date_fetched'] = pd.to_datetime(self.df.date_fetched)
        self.df['planned_expiration_date'] = pd.to_datetime(
            self.df.planned_expiration_date)

        self.df = self.df[(self.df.planned_expiration_date <
                           self.df.date_fetched) &
                          (self.df.posted_date > self.min_date)]

        self.df['days_available'] = ((self.df.planned_expiration_date -
                                     self.df.posted_date)/np.timedelta64
                                     (1, 'D')).round().astype(int)

        self.df['end_date'] = pd.to_datetime(self.df.funded_date)
        self.df['end_date'][self.df.end_date.isnull()] = \
            self.df.planned_expiration_date
        self.df['days_on_kiva'] = ((self.df.end_date - self.df.posted_date) /
                                   np.timedelta64(1, 'D'))

    def transform_labels(self):
        '''
        labels loans as either expired or funded. Drops the current status
        info and drops the tiny number of loans which were refunded
        (withdrawn from kiva without being funded or expiring)
        '''
        self.df['expired'] = self.df.status == 'expired'
        self.df = self.df[self.df.status != 'refunded']

    def transform_df(self):
        '''
        Transforms the dataframe using the 6 functions above
        Drops columns not used in model
        '''
        self.df = self.df.drop_duplicates(['id'])
        self.df = self.df.set_index('id')
        self.transform_dates()
        self.get_desc()
        self.payment_terms()
        self.borrower_info()
        self.transform_themes()
        self.transform_labels()
        self.df.drop(self.drop_cols, axis=1, inplace=True)

    def setup_sql(self, user, pw, db='kiva', host='localhost', port='5432'):
        '''
        sets up sql connection for exporting and loading from sql
        must be run before export_to_sql, load_from_sql, or merge_db
        '''
        self.sql['db'] = db
        self.sql['user'] = user
        self.sql['pw'] = pw
        self.sql['host'] = host
        self.sql['port'] = str(port)
        self.tables = []
        engstr = 'postgresql://%s:%s@%s:%s/%s' % (user, pw, host, port, db)
        self.sql_engine = create_engine(engstr)

    def run_query(self):
        '''
        Executes self.query. Used by export_to_sql, merge_db, competing_loans
        '''
        print self.query
        conn = psycopg2.connect(dbname=self.sql['db'], user=self.sql['user'],
                                host=self.sql['host'], password=self.sql['pw'])
        c = conn.cursor()
        c.execute(self.query)
        conn.commit()
        conn.close()
        self.query = ''


    def export_to_sql(self, table):
        '''
        input name of table to insert the pandas dataframe self.df
        using the sql db, user, and pw input by the setup sql function
        '''
        self.query += 'DROP TABLE if exists %s' % table
        self.run_query()
        self.df.to_sql(table, self.sql_engine)
        self.tables.append(table)

    def load_from_sql(self, table=None, where='', date_range=None, cols='*'):
        '''
        optional input table to load from, default is self.table
        optional input cols to load from sql as string, default is *
        optional input date_range as tuple in '2014-01-01' format
        optional input 'WHERE' query if you don't select by date range
        '''
        if not table:
            table = self.tables[-1]
        if date_range:
            where = '''WHERE posted_date > '%s'
                       AND posted_date <  '%s' ''' % date_range
        query = 'SELECT %s FROM %s %s;' % (cols, table, where)
        self.df = pd.read_sql_query(query, self.sql_engine)

    def merge_db(self):
        '''
        merge multiple sql dbs into 1 db
        '''
        self.query += '''DROP TABLE IF EXISTS merged;
            CREATE TABLE merged AS ''' + \
            'UNION '.join('(SELECT * FROM %s) ' % tab for
                          tab in self.tables) + '; ' + \
            ''.join(' DROP TABLE %s; ' % tab for tab in self.tables)
        self.run_query()
        self.tables = ['merged']

    def competing_loans(self, new_table='loans'):
        '''
        add column competing_loans which calculates total number of loans
        fundraising on kiva.org when each loan was posted
        '''
        self.query += '''DROP TABLE IF EXISTS %s;
            CREATE VIEW supply as
            SELECT count(1) competing_loans, a.posted_date
            FROM
                (SELECT DISTINCT posted_date FROM merged
                WHERE posted_date >
                    (SELECT min(posted_date) + INTERVAL '30 days'
                    FROM merged)) a
            JOIN
                (SELECT posted_date, end_date FROM merged) b
            ON a.posted_date > b.posted_date AND a.posted_date < b.end_date
            GROUP BY a.posted_date;

            CREATE TABLE %s as
            SELECT * FROM merged LEFT JOIN supply using (posted_date);
            DROP VIEW supply;
            DROP TABLE merged; ''' % (new_table, new_table)

        self.run_query()
        self.tables = [new_table]

    def sql_pipeline(self, address, user, pw, batch=60, competing_loans=True,
                     table_name='loans'):
        '''
        imports, transforms, and loads loan data into sql.
        address is folder containing json files from kiva api.
        batch is the number of json files to transform at a time.
        default 60 files = 30,000 kiva loan records which is about 128 MB.
        '''
        filelist = glob.glob(address + "/*.json")
        self.setup_sql(user, pw)
        for n in xrange(0, (len(filelist)-1)/batch+1):
            self.import_loans(files=filelist[n*batch:(n+1)*batch])
            self.transform_df()
            if self.df.shape[0]:
                self.export_to_sql('temp_' + str(n))
        self.merge_db()
        if competing_loans:
            self.competing_loans(table_name)
