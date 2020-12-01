import finnhub
import pandas as pd
import csv
#import yfinance as yf
#data = yf.download("AAPL", start="2017-01-01", end="2017-04-30",interval="15m")
#print(data)
symbols = []
count = 0
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
print(symbols)
# Setup client
finnhub_client = finnhub.Client(api_key="bupp3dn48v6tbk6ddu2g")

for s in symbols:
    res = finnhub_client.stock_candles(s,'D',1451624400 ,1605106902) #2016 - today
    #Convert to Pandas Dataframe and save as csv file
    (pd.DataFrame(res)).to_csv(rf'C:\final project data\{s}.csv',index = False)
    count += 1
    #if count % 20 == 0:
     #   time.sleep(1800)