import pandas as pd
import numpy as np
import scipy.stats as stats
import dataDownload
import matplotlib.pyplot as plt
import datetime as dt
#symbols=['AAL','AAP','ABBV','ABC','ABMD','ABT','ACN','AAPL','GOOGL']
symbols=dataDownload.createSymbolsList()
print(symbols)

#create table of input data
def createTable():
    columns =['t-13','t-12','t-11','t-10','t-9','t-8','t-7','t-6','t-5','t-4','t-3','t-2',
              'd1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','isJanuary']
    df = pd.DataFrame( columns = columns, index=symbols)
    return df

#return the daily return from the input df row
def dayReturn(row):
    open=row['Open']
    close=row['Close']
    return (close/open)-1
#return the monthly return from the input df rows
def monthReturn(dataframe):
    open=dataframe['Open'].iloc[0]
    close=dataframe['Close'].iloc[-1]
    return (close/open)-1
#return df rows of the requested month and year
def oneMonth(df,month,year):
    y=df['Date'].dt.year.between(year,year)
    #Aug= dataframe['Date'].dt.month.between(8,8)
    df_year = df.loc[y]
    m=df_year['Date'].dt.month.between(month,month)
    df_month = df_year.loc[m]
    return  df_month
#return a valid value of month
def validMonth(val):
    if(val==0 or 12):
        return 12

    return(val%12)

def generateInputData(inputTable,month,year):
    columns =['t+1']
    Yverctor = pd.DataFrame( columns = columns, index=symbols)
    #####not all stocks started in 2000 so when we see stock that dont have data in the year or month we need we skip it
    badYear=False
    for s in symbols:
        print("start returns"+s,month,year)
        dataframe = pd.read_csv(rf'C:\final project data\{s}.csv')
        dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')#change date format
        j=13
        y=year
        for i in range(12):
            if month-j<=0:
                y=year-1;
            else:
                y=year
            if(dataframe['Date'].dt.year[0]>y):
                badYear=True#skip to the next stock
                break
            elif(dataframe['Date'].dt.month[0]>validMonth(month-j)):
                badYear=True
                break
            currMonth=oneMonth(dataframe,validMonth(month-j),y)
            #print(s,validMonth(month-j),y)
            x=monthReturn(currMonth)
            inputTable.loc[s][i]=x
            j-=1

        if(badYear):
            badYear=False
            continue#skip to the next stock
        #last 20 days
        j=0
        currMonth=oneMonth(dataframe,month,year)
        rowCount= currMonth.shape[0]
        for i in range(12,32,1):
            if(rowCount <= j or 20<=j):
                print(j)
                break
            x=dayReturn(currMonth.iloc[j])
            inputTable.loc[s][i]=x
            j+=1
        #creating monthly returns vector for month+1
        y=year
        if month+1>12:
            y=year+1;
        nextMonth=oneMonth(dataframe,validMonth(month+1),y)
        #print(nextMonth)
        x=monthReturn(nextMonth)
        Yverctor.loc[s][0]=x
    if month==1:
        inputTable.iloc[:,32]=1
    else:
        inputTable.iloc[:,32]=0
    #delete empty rows
    #print(Yverctor.shape[0])
    Yverctor.dropna(subset = ["t+1"], inplace=True)
    #print(Yverctor.shape[0])
    Yverctor.to_csv(rf'Next month returns\NextMonthReturnsForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    inputTable.to_csv(rf'Data\inputDataForMonth-{str(month)},Year-{str(year)}.csv',index = False)
    return inputTable


def CumulativeReturn(row,j):
    res=1
    x=0
    if j<12:
        for i in range(j+1):
             x=1+row.iloc[i]
             res*=x
    else:
        for i in range(12,j+1):
            x=1+row.iloc[i]
            res*=x

    return res-1

def generateInputCumulativeReturn(month,year):
    s='inputDataForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    outputTable= createTable()
    for i in range(outputTable.shape[0]):
        if(inputTable.iloc[i][0]==None):
            continue
        for j in range(32):
            if j<12 and j>=0:
                outputTable.iloc[i][j]=CumulativeReturn(inputTable.iloc[i],j)
            elif j>=12:
                outputTable.iloc[i][j]=CumulativeReturn(inputTable.iloc[i],j)
    outputTable.iloc[:,32]=inputTable.iloc[:,32]
    #delete empty rows
    print(outputTable.shape[0])
    outputTable.dropna(subset = ["t-13"], inplace=True)
    print(outputTable.shape[0])#print how many stocks left
    outputTable.to_csv(rf'Data\CumulativeReturnForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    return outputTable

def generateZscore(month,year):
    s='CumulativeReturnForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    outputTable= inputTable.copy()
    for i in range(1,outputTable.shape[1]-1):
        arr=stats.zscore(inputTable.iloc[:,i])
        for j in range(outputTable.shape[0]-1):
                outputTable.iloc[j][i]=arr[j]
    #outputTable.iloc[:,32]=inputTable.iloc[:,32]
    #fill in indecator
    if month==1:
        outputTable.iloc[:,32]=1
    else:
        outputTable.iloc[:,32]=0
    outputTable.to_csv(rf'Data\ZscoreForMonth-{str(month)},Year-{str(year)}.csv',index = False)
    return outputTable

def calculateMedian(inputVector):
    arr = np.array(inputVector.iloc[:,1])
    length = len(arr)
    arr.sort()

    if length % 2 == 0:
        median1 = arr[length//2]
        median2 = arr[length//2 - 1]
        median = (median1 + median2)/2
    else:
        median = arr[length//2]

    return median

def generateInputClassification(month,year):
    s='NextMonthReturnsForMonth-'+str(month)+',Year-'+str(year)
    inputVector = pd.read_csv(rf'Next month returns\{s}.csv')
    median=calculateMedian(inputVector)
    #create vector to classify examples with returns below the median
    #as belonging to class 1 and those with returns above
    #the median to class 2.
    columns =['class']
    rows=inputVector.iloc[:,0].values
    Yverctor = pd.DataFrame( columns = columns, index=rows)
    for i in range(Yverctor.shape[0]):
        if inputVector.iloc[i]['t+1']<median:
            Yverctor.iloc[i]['class']=0
        else:
            Yverctor.iloc[i]['class']=1

    Yverctor.to_csv(rf'Data\classForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    return Yverctor


#appendMonthsVector(2,2017)
'''
Table = pd.read_csv(rf'FINAL DATA\classForMonth-1,Year-2018.csv')
Table=Table.drop(Table.columns[[0]],1)
print(Table)
df=createTable()
ReturnsForMonth=generateInputData(df,10,2020)
outputTable =generateInputCumulativeReturn(10,2020)
ReturnsForMonth.loc['GOOGL'][0:12].plot(kind='bar')
'''
#X= generateZscore(10,2020)
#ReturnsForMonth.iloc[7][0:12].plot(kind='bar')
#plt.show()
#ReturnsForMonth.loc['AAL'].plot(kind='bar')#iloc[0,0:11].plot()
#outputTable =generateInputCumulativeReturn(2,2017)
#X= generateZscore(2,2017)
#y=generateInputClassification(2,2017)


#############################################################################
#inputTable=createTable()
#generateInputData(inputTable,3,2001)
#x=generateInputCumulativeReturn(3,2001)
#generateZscore(3,2001)
#generateInputClassification(3,2001)