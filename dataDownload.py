import finnhub
import pandas as pd
import csv


#step #1: download data from finnhub and create csv file in c: folder
symbols = []
count = 0
history_symbols=[]
list=['LDW','JDSU','BS','QTRN','TSG','LUK','FDC','TSO','JEC','DJ','LO','FRE','FNM']
def createSymbolsList():

    with open('sp500_history.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                symbols.append(row[5])
                line_count += 1
    #for i in range(250):
    #    symbols.pop(0)
            ###############################################
    '''

        #delete problematic stocks from symbol list
        symbols.remove("CTL");
        symbols.remove("CNP");
        symbols.remove("DXC");
        symbols.remove("MSCI");
        symbols.remove("NI");
        symbols.remove("ETFC");
        ##############################################3

    df=pd.read_csv("sp500_history.csv")
    history_symbols=df['value']
    #print(history_symbols.head())
    '''
    return symbols



# Setup client
finnhub_client = finnhub.Client(api_key="bupp3dn48v6tbk6ddu2g")

'''
history_symbols=createSymbolsList()
for s in history_symbols:
    if(count>325):
        try:
            res = finnhub_client.stock_candles(s,'D',315532800 ,1609459200) #1980 - 2021
            #Convert to Pandas Dataframe and save as csv file
            (pd.DataFrame(res)).to_csv(rf'C:\final project data 1980-2021\{s}.csv',index = True)
        except:
            print(res)
            list.append(s)
    count += 1
    print(s+"-"+str(count))
    #if count % 20 == 0:
     #   time.sleep(1800)
    pd.DataFrame(list).to_csv("badSymbols.csv")
    '''