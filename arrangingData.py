import pandas as pd

import argparse
from datetime import date
import tzlocal

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()
#create list of the stocks names(symbols)
symbols = ['SPY']
print(symbols)

for s in symbols:
    args = parse_args()
    datapath = (rf'Data\{s}.csv')
    #Simulate the header row isn't there if noheaders requested
    skiprows = 1 if args.noheaders else 0
    header = None if args.noheaders else 0
    dataframe = pd.read_csv(datapath)
    if not args.noprint:
        clist = ['t','o','h','l','c','v','s']
        df=dataframe.copy()
        df=df[clist]

    datesArr = pd.Series(df['t'].values.astype(float))
    local_timezone = tzlocal.get_localzone() # get pytz timezone
    i=0
    for d in datesArr:
        local_time = date.fromtimestamp(d)#, local_timezone).strftime('%d-%m-%Y')
        df['t'].iloc[i]=pd.to_datetime(local_time)
        df['s'].iloc[i]=0
        i+=1
    df.rename(columns={'t': 'Date', 'o': 'Open','h':'High','l':'Low','c':'Close','v':'Volume','s':'openinterest'},inplace=True)
    df.set_index('Date', inplace = True)
    df.to_csv(rf'Data\{s}.csv',index = True)


##test

