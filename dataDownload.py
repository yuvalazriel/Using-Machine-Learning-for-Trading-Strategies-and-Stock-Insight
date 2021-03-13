import finnhub
import pandas as pd
import csv


#step #1: download data from finnhub and create csv file in c: folder
symbols = []
count = 0

def createSymbolsList():
    with open('constituents_csv.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                symbols.append(row[0])
                line_count += 1
    #for i in range(250):
    #    symbols.pop(0)
            ###############################################
        #delete problematic stocks from symbol list
        symbols.remove("CTL");
        symbols.remove("CNP");
        symbols.remove("DXC");
        symbols.remove("MSCI");
        symbols.remove("NI");
        symbols.remove("ETFC");
        ##############################################3
    return symbols



# Setup client
finnhub_client = finnhub.Client(api_key="bupp3dn48v6tbk6ddu2g")

for s in symbols:
    #if(count>338):
    res = finnhub_client.stock_candles(s,'D',946684800 ,1609459200) #2000 - 2021
    #Convert to Pandas Dataframe and save as csv file
    (pd.DataFrame(res)).to_csv(rf'C:\final project data\{s}.csv',index = True)
    count += 1
    print(s+"-"+str(count))
    #if count % 20 == 0:
     #   time.sleep(1800)