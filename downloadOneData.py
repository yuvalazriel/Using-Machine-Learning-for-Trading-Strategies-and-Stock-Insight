import finnhub
import pandas as pd


finnhub_client = finnhub.Client(api_key="bupp3dn48v6tbk6ddu2g")
res = finnhub_client.stock_candles('SPY','D',1451624400 ,1605106902) #2016 - today
#Convert to Pandas Dataframe and save as csv file
(pd.DataFrame(res)).to_csv(rf'C:\Users\inbar\Documents\שנה ד\Final Project\Data\SPY.csv',index = False)

