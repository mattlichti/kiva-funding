import pandas as pd
import numpy as np
import json
import glob
import os
import random

min_date = '2012-01-25' # when kiva expiration policy fully implemented
max_date = '2014-12-22' # last day when all loans in dataset could expire

def import_loans(address):
    '''
    Input address of folder full of json files
    Output python list of kiva loans
    ''' 
    lst = []
    os.chdir(address)
    for file in glob.glob("*.json"):
        f = open(file)
        dic = json.loads(f.readline())
        lst +=(dic['loans'])
    return lst

def build_df(lst):
    '''
    converts list of kiva loans into df, drops the data that was generated
    after loan was funded, and extracts most useful features
    '''

    df = pd.DataFrame(lst)
    droplist = ['basket_amount','currency_exchange_loss_amount','delinquent',
                'paid_date', 'paid_amount', 'journal_totals', 'payments']
    droplist += ['lender_count', 'funded_date','funded_amount']
    droplist += ['translator', 'video']
    df.drop(droplist,axis=1,inplace=True)

    df = get_country(df)
    df = get_desc(df)
    df = payment_terms(df)
    df = borrower_info(df)
    df = expiration_date(df)

    # throwing away image template info because they only have one template
    df['image'] = df.image.map(lambda x: x['id'])

    df = df[(df.posted_date < max_date) & (df.posted_date > min_date)]
    return df

def get_country(df):
    ldf = pd.DataFrame(list(df.location))
    df['country'] = ldf['country_code']
    df.drop('location',axis=1,inplace=True)
    return df  

def get_desc(df):
    ddf = pd.DataFrame(list(df.description))
    tdf = pd.DataFrame(list(ddf.texts))
    df['description'] = tdf['en']
    return df

def payment_terms(df):
    terms = pd.DataFrame(list(df.terms))
    df['repayment_interval'] = terms['repayment_interval']
    df['repayment_term'] = terms['repayment_term']
    curr_loss = pd.DataFrame(list(terms.loss_liability))
    df['currency_loss'] = curr_loss.currency_exchange
    df.drop('terms',axis=1,inplace=True)
    return df

def borrower_info(df):
    df['group_size'] = df.borrowers.map(lambda x: len(x))
    df['gender'] = df.borrowers.map(lambda x: x[0]['gender'])
    df.drop('borrowers',axis=1,inplace=True)
    return df

def expiration_date(df):
    '''
    calculates days loan is available on site until planned expiration
    '''

    df['posted_date'] = pd.to_datetime(df.posted_date)
    df['planned_expiration_date'] = pd.to_datetime(df.planned_expiration_date)
    df['days_available'] = ((df.planned_expiration_date - df.posted_date)/
                            np.timedelta64(1, 'D')).round().astype(int)
    df.drop('planned_expiration_date',axis=1,inplace=True)
    return df

def get_start_date(df):
    print df[df.planned_expiration_date.isnull()].posted_date.max()

def dump(df, fname):
    loans = df.to_json()
    os.chdir(address + 'dumps')
    with open(fname, 'w') as outfile:
        json.dump(loans, outfile)

def clean_and_dump(folders):
    for folder in folders:
        lst = import_loans(folder)
        df = build_df(lst)
        dump(df,folder +'.json')

def load_cleaned(lst,drops = [],reindex=False):
    os.chdir(address + 'dumps')
    dfs = []
    for jsn in lst:
        f = open(jsn + '.json')
        dic = json.loads(f.read())
        f.close()
        df = pd.read_json(dic)
        if drops != []:
            df.drop(drops,axis=1,inplace=True)
        dfs.append(df)
    # return pd.concat(dfs,axis=0)
    df = pd.concat(dfs,axis=0)
    df.posted_date = pd.to_datetime(df.posted_date*10**6)
    if reindex:
        df = df.drop_duplicates(['id'])        
        df = df.set_index('id')
    return df

def condense_jsons(file_list):
    df = load_cleaned(file_list, drops = ['image', 'name', 'partner_id'],
                                 reindex=True)
    dump(df,'everything.json')