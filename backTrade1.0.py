import backtrader
import argparse
from strategies import TestStrategy
import pandas
from datetime import date
import tzlocal  # $ pip install tzlocal

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()

#def runstrat():

args = parse_args()
# Create a cerebro entity
cerebro = backtrader.Cerebro(stdstats=False)
cerebro.broker.setcash(1000000.0)
cerebro.addstrategy(TestStrategy)
cerebro.addsizer(backtrader.sizers.FixedSize, stake=1000)

# Get a pandas dataframe
datapath = ('AAPL2.csv')

# Simulate the header row isn't there if noheaders requested
skiprows = 1 if args.noheaders else 0
header = None if args.noheaders else 0

dataframe = pandas.read_csv(datapath)#,
                            #skiprows=skiprows,
                           #header=None,
                          #parse_dates=True,
                          #index_col=0)

if not args.noprint:
    print('--------------------------------------------------')
    clist = ['t','o','h','l','c','v','s']
    df=dataframe.copy()
    df=df[clist]
    #df.replace({'A': 'ok'}, {'A': '0'}, regex=True)
    print(df.head(5))
    print('--------------------------------------------------')

datesArr = pandas.Series(df['t'].values.astype(float))
local_timezone = tzlocal.get_localzone() # get pytz timezone
i=0
for s in datesArr:
    local_time = date.fromtimestamp(s)#, local_timezone).strftime('%d-%m-%Y')
    df['t'].iloc[i]=pandas.to_datetime(local_time)
    df['s'].iloc[i]=0
    i+=1

df.rename(columns={'t': 'Date', 'o': 'Open','h':'High','l':'Low','c':'Close','v':'Volume','s':'openinterest'},inplace=True)
print("this is the arr after change")
print(df.head(5))
df.set_index('Date', inplace = True)
print("this is the arr after change 2")
print(df.head(5))
df.to_csv(rf'Data\AAPL.csv',index = True)

# Create a Data Feed

data = backtrader.feeds.PandasData(dataname=df)
     # dataname='ORCL.csv',
    # Do not pass values before this date
     #fromdate=datetime.datetime(2016, 1, 1),
    # Do not pass values after this date
      #todate=datetime.datetime(2016, 12, 31),
  #reverse=False)
cerebro.adddata(data)
#cerebro.addstrategy(TestStrategy)
#cerebro.addsizer(backtrader.sizers.FixedSize, stake=1000)




#if __name__ == '__main__':
 #   runstrat()

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.run()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()
