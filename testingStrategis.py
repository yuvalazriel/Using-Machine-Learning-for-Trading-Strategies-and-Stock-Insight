import backtrader
from strategies import TestStrategy
import pandas as pd
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
#pip3 install backtrader_plotting
#import matplotlib.pyplot as plt

plotconfig = {
    'id:ind#0': dict(
       subplot=True,
   ),
}
b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo(), plotconfig=plotconfig)

def create_stock_sp500_ratio_csv(stockName):
    datapath = (rf'Data\{stockName}.csv')
    dataframeStock = pd.read_csv(datapath)
    datapath = (rf'Data\SPY.csv')
    dataframeSPY = pd.read_csv(datapath)
    #crate copy
    df=dataframeSPY.copy()
    for i in range(df.shape[0]):# gives number of row count
       # df['Date'].iloc[i]-=dataframeStock['Date'].iloc[i]
        df['Open'].iloc[i]-=dataframeStock['Open'].iloc[i]
        df['High'].iloc[i]-=-dataframeStock['High'].iloc[i]
        df['Low'].iloc[i]-=dataframeStock['Low'].iloc[i]
        df['Close'].iloc[i]-=dataframeStock['Close'].iloc[i]
        df['Volume'].iloc[i]-=dataframeStock['Volume'].iloc[i]

    fileName=stockName+'_sp500_ratio'
    df.to_csv(rf'Data\{fileName}.csv',index = True)

    return fileName+'.csv'

#Strategy=TestStrategy, datapath= full path, setcash= how much money wo want to invest, sizers= num of stocks to buy
def run_Strategy(Strategy,datapath,setcash,sizers):

    cerebro = backtrader.Cerebro(stdstats=False)
    cerebro.broker.setcash(setcash)
    cerebro.addstrategy(Strategy)
    cerebro.addsizer(backtrader.sizers.FixedSize, stake=sizers)
    dataframe = pd.read_csv(datapath,parse_dates=['Date'])
    print('--------------------------------------------------')
    print(dataframe.head(5))
    df=dataframe.copy()
    df.set_index('Date', inplace = True)
    print('--------------------------------------------------')
    print(df.head(5))
    data = backtrader.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    #cerebro.plot()
    return cerebro


datapath=create_stock_sp500_ratio_csv('A')

cerebro_regular_stock=run_Strategy(TestStrategy,'Data\A.csv',100000,100)
datapath = 'Data\\' + datapath
print("######################################################################################################################")
cerebro_SPY_ratio_stock=run_Strategy(TestStrategy,datapath,100000,100)

cerebro_regular_stock.plot(b)
#cerebro_SPY_ratio_stock.plot(b)