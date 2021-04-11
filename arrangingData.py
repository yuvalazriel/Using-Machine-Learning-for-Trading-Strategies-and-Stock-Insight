import pandas as pd
import dataDownload
import argparse
from datetime import date
import tzlocal
import csv
#step #2:  change csv file in c: folder to the right format of table
def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()
badSymbols=[]
def createbadSymbolsList():

    with open('badSymbols.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                badSymbols.append(row[1])
                line_count += 1

    return badSymbols
#create list of the stocks names(symbols)
symbols=dataDownload.createSymbolsList()
print(len(symbols))
badSymbols=createbadSymbolsList()
#print(badSymbols)
for s in badSymbols:
    symbols.remove(s)
cnt=0
print(len(symbols))
print(symbols)
for s in symbols:
    if(cnt>734):
        args = parse_args()
        datapath = (rf'C:\final project data 1980-2021\{s}.csv')#final project data 1980-2021
        #Simulate the header row isn't there if noheaders requested
        skiprows = 1 if args.noheaders else 0
        header = None if args.noheaders else 0
        dataframe = pd.read_csv(datapath)
        print("start:"+s+"-"+str(cnt))
        if not args.noprint:
            clist = ['t','o','h','l','c','v','s']
            df=dataframe.copy()
            df=df[clist]
#change timestamp to date in YYYY-MM-DD format
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
        df.to_csv(rf'C:\final project data 1980-2021\{s}.csv',index = True)
    #else:
        #break
    print("end:"+s+"-"+str(cnt))
    cnt+=1

