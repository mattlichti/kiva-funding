import pandas as pd
import matplotlib.pyplot as plt
import data_pipeline
import build_model
import seaborn
import numpy as np
import sys


class Plots(object):

    def __init__(self, user, pw, db='kiva', table='loans', host='localhost',
                 port='5432', dates=('2012-04-01', '2015-04-01')):
        '''
        Input Postgresql username & pw
        makes univariate and multivariate plots of kiva loan features
        '''
        self.dates = dates
        self.table = table
        self.pipe = data_pipeline.Pipeline()
        self.pipe.setup_sql(user, pw, db=db, host=host, port=port)

    def competing_loans(self):
        '''
        plots expiration rate by number of other loans fundraising on kiva.org
        when the loan was posted
        '''
        self.pipe.load_from_sql(cols='competing_loans, gender, expired',
                                date_range=self.dates, table=self.table)
        df = self.pipe.df
        edges = np.concatenate(([df.competing_loans.min()],
                               np.arange(1000, 8000, 1000),
                               [df.competing_loans.max()])).astype(int)
        rate, men_rate, women_rate, interval = [], [], [], []

        for i in xrange(8):
            rate.append(df[(df.competing_loans > edges[i]) &
                           (df.competing_loans < edges[i+1])
                           ].expired.mean()*100)
            men_rate.append(df[(df.competing_loans > edges[i]) &
                               (df.competing_loans < edges[i+1]) &
                               ~df.gender].expired.mean()*100)
            women_rate.append(df[(df.competing_loans > edges[i]) &
                                 (df.competing_loans < edges[i+1]) &
                                 df.gender].expired.mean() * 100)
            interval.append('%s-%s' % (edges[i], edges[i+1]))

        fig, ax = plt.subplots()
        index = np.arange(len(men_rate))
        bar_width = 0.28
        plt.bar(index, rate, bar_width, alpha=.8, color='g', label='Overall')
        plt.bar(index + bar_width, men_rate, bar_width, alpha=.8,
                color='b', label='Men')
        plt.bar(index + 2 * bar_width, women_rate, bar_width, alpha=.6,
                color='r', label='Women')
        plt.xlabel('Number of loans fundraising when each loan was posted')
        plt.ylabel('Percent of loans that expired')
        plt.title('Expiration rate by gender and number of competing loans')
        plt.xticks(index + bar_width*1.5, interval)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def month(self):
        '''
        plots expiration rate by repayment interval
        '''
        self.pipe.load_from_sql(cols='posted_date, expired',
                                date_range=self.dates, table=self.table)
        df = self.pipe.df
        dates, exp = [], []
        months = ['Jan', 'Feb', 'Mar', "April", 'May', 'June', 'July',
                  'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i in xrange(36):
            start_date = '201%s-%s-01' % ((i+3)/12 + 2, (i+3) % 12 + 1)
            end_date = '201%s-%s-01' % ((i+4)/12 + 2, (i+4) % 12 + 1)

            exp.append(round(df[(df.posted_date > start_date) &
                                (df.posted_date < end_date)
                                ].expired.mean()*100, 1))
            dates.append('%s 201%s' % (months[(i+3) % 12], (i+3)/12 + 2))

        fig, ax = plt.subplots()
        index = np.arange(len(exp))
        plt.bar(index, exp, color='g')
        plt.xticks(index + .5, dates, rotation='vertical')
        plt.xlabel('Month')
        plt.ylabel('Percent of loans that expired')
        plt.title('Expiration rate by month')
        plt.tight_layout()
        plt.show()

    def gender(self):
        '''
        Plots expiration rate by gender
        '''
        self.pipe.load_from_sql(cols='gender, expired', date_range=self.dates,
                                table=self.table)
        df = self.pipe.df
        male = df[~df.gender].expired.mean()
        female = df[df.gender].expired.mean()

        plt.bar([0, 1], [male, female])
        plt.xticks([0.5, 1.5], ['Male', 'Female'])
        plt.ylabel('Expiration Rate')
        plt.show()

    def payment_int(self):
        '''
        plots expiration rate by repayment interval
        '''
        self.pipe.load_from_sql(cols='repayment_interval, expired',
                                date_range=self.dates, table=self.table)
        df = self.pipe.df
        monthly = df[df['repayment_interval'] == 'Monthly'].expired.mean()
        irreg = df[df['repayment_interval'] == 'Irregularly'].expired.mean()
        end = df[df['repayment_interval'] == 'At end of term'].expired.mean()

        plt.bar([0, 1, 2], [monthly, irreg, end])
        plt.xticks([0.5, 1.5, 2.5], ['Monthly', 'Irregularly', 'End of Term'])
        plt.ylabel('Expiration Rate')
        plt.xlabel('Repayment Schedule')
        plt.show()

if __name__ == '__main__':
    '''
    Input postgres username and password. Optional: input postgres
    db name, table name, host, port, date range
    ex: python rplots.py username password
    '''
    p = Plots(*sys.argv[1:])
    p.competing_loans()
    p.month()
