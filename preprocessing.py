import pandas as pd
import datetime as dt
import numpy as np
import scipy.stats as stats


symbols=['AAL','AAP','ABBV','ABC','ABMD','ABT','ACN']

def createTable():
    columns =['t-13','t-12','t-11','t-10','t-9','t-8','t-7','t-6','t-5','t-4','t-3','t-2',
              'd1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','isJanuary']
    #create table of input data
    #constituents_csv = pd.read_csv('constituents_csv.csv')
    #symbols=[constituents_csv.Symbol]# company symbol is the index
    #print(symbols)
    #symbols=['A','AAL','AAP','ABBV','ABC','ABMD','ABT','ABT']
    df = pd.DataFrame( columns = columns, index=symbols)
    return df

#return the daily return - should return in percentage?
def dayReturn(row):
    open=row['Open']
    close=row['Close']
    return (close/open)-1

def monthReturn(dataframe):
    open=dataframe['Open'].iloc[0]
    close=dataframe['Close'].iloc[-1]
    return (close/open)-1

def oneMonth(df,month,year):
    y=df['Date'].dt.year.between(year,year)
    #Aug= dataframe['Date'].dt.month.between(8,8)
    df_year = df.loc[y]
    m=df_year['Date'].dt.month.between(month,month)
    df_month = df_year.loc[m]
    return  df_month

def validMonth(val):
    if(val==0):
        return 12
    return(val%12)

def generateInputData(inputTable,month,year):
    columns =['t+1']
    Yverctor = pd.DataFrame( columns = columns, index=symbols)
    for s in symbols:
        dataframe = pd.read_csv(rf'Data\{s}.csv')
        dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')
        #row= df.filter(like = s, axis=0)
        j=13
        y=year
        for i in range(12):
            if month-j<0:
                y=year-1;
            currMonth=oneMonth(dataframe,validMonth(month-j),y)
            x=monthReturn(currMonth)
            inputTable.loc[s][i]=x
            j-=1
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
    Yverctor.to_csv(rf'Data\NextMonthReturnsForMonth-{str(month)},Year-{str(year)}.csv',index = False)
    inputTable.to_csv(rf'Data\inputDataForMonth-{str(month)},Year-{str(year)}.csv',index = False)


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
        for j in range(32):
            if j<12 and j>=0:
                outputTable.iloc[i][j]=CumulativeReturn(inputTable.iloc[i],j)
            elif j>=12:
                outputTable.iloc[i][j]=CumulativeReturn(inputTable.iloc[i],j)
    outputTable.iloc[:,32]=inputTable.iloc[:,32]
    outputTable.to_csv(rf'Data\CumulativeReturnForMonth-{str(month)},Year-{str(year)}.csv',index = False)
    return outputTable

def generateZscore(month,year):
    s='CumulativeReturnForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    outputTable= inputTable.copy()
    for i in range(outputTable.shape[1]-1):
        arr=stats.zscore(inputTable.iloc[:,i])
        for j in range(outputTable.shape[0]):
                outputTable.iloc[j][i]=arr[j]
    outputTable.iloc[:,32]=inputTable.iloc[:,32]
    outputTable.to_csv(rf'Data\ZscoreForMonth-{str(month)},Year-{str(year)}.csv',index = False)
    return outputTable

def calculateMedian(inputVector):
    arr = np.array(inputVector.iloc[:,0])
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
    inputVector = pd.read_csv(rf'Data\{s}.csv')
    median=calculateMedian(inputVector)
    #create vector to classify examples with returns below the median
    #as belonging to class 1 and those with returns above
    #the median to class 2.
    columns =['class']
    Yverctor = pd.DataFrame( columns = columns, index=symbols)
    for i in range(Yverctor.shape[0]):
        if inputVector.iloc[i][0]<median:
            Yverctor.iloc[i][0]=1
        else:
            Yverctor.iloc[i][0]=2

    Yverctor.to_csv(rf'Data\classForMonth-{str(month)},Year-{str(year)}.csv',index = True)


#df=createTable()
#generateInputData(df,2,2017)
#outputTable =generateInputCumulativeReturn(2,2017)
#outputTable= generateZscore(2,2017)
generateInputClassification(2,2017)
