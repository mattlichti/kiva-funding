import pandas as pd
import json
import glob
import os

# address = "/home/matt/gdrive/final_project/Fundraising-Success/data/loans/"
address = "/Users/datascientist/matt/project/loans/"


def import_loans(folder,address=address):
    lst = []
    os.chdir(address + folder)
    for file in glob.glob("*.json"):
        f = open(file)
        dic = json.loads(f.readline())
        lst +=(dic['loans'])
    return lst

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

def build_df(lst):
    df = pd.DataFrame(lst)
    droplist = ['basket_amount','currency_exchange_loss_amount','delinquent', 'paid_date', 'paid_amount', 'journal_totals', 'payments']
    droplist += ['lender_count', 'funded_date','funded_amount']
    droplist += ['translator', 'video']
    df.drop(droplist,axis=1,inplace=True)

    df['posted_date'] = pd.to_datetime(df.posted_date)
    df['planned_expiration_date'] = pd.to_datetime(df.planned_expiration_date)

    df = get_country(df)
    df = get_desc(df)
    df = payment_terms(df)
    df = borrower_info(df)
    return df

def get_start_date(df):
    print df[df.planned_expiration_date.isnull()].posted_date.max()


def dump(df, fname):
    loans = df.to_json()
    os.chdir(address + 'dumps')
    with open(fname, 'w') as outfile:
        json.dump(loans, outfile)

def clean_and_dump(folder):
    lst = import_loans(folder)
    df = build_df(lst)
    dump(df,folder +'.json')

def load_cleaned(lst):
    os.chdir(address + 'dumps')
    dfs = []
    for jsn in lst:
        f = open(jsn + '.json')
        dic = json.loads(f.read())
        f.close()
        df = pd.read_json(dic)
        dfs.append(df)
    return pd.concat(dfs,axis=0)