import pandas as pd
import numpy as np
import scipy.stats as stats
import dataDownload
import symboleListCreate
import matplotlib.pyplot as plt
import datetime as dt
'''
symbols = symboleListCreate.listSymbolsCreate()
symbols.remove("ABK")
symbols.remove("CBE")
symbols.remove("CEG")
symbols.remove("EP")
symbols.remove("SAI")
'''
#create table of input data
def createTable(list):
    columns =['t-13','t-12','t-11','t-10','t-9','t-8','t-7','t-6','t-5','t-4','t-3','t-2',
              'd1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17','d18','d19','d20','isJanuary']
    df = pd.DataFrame(columns=columns, index= list)
    return df

#return the daily return from the input df row
def dayReturn(row1, row2):
    return (row2['Open']/row1['Open']) - 1

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
    return df_month

#return a valid value of month
def validMonth(val):
    if val == 0 or val == 12 or val == -12:
        return 12
    return val % 12


def nextMonthReturns(month, year, list):
    #print(list)
    columns =['t+1']
    Yverctor = pd.DataFrame(columns=columns, index=list)
    y=year
    if month+1>12:
        y=year+1

    for s in list:
        dataframe = pd.read_csv(rf'final project data 1980-2021\{s}.csv')
        dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')#change date format
        nextMonth=oneMonth(dataframe,validMonth(month+1),y)
        print(nextMonth,s)
        x=monthReturn(nextMonth)
        Yverctor.loc[s][0]=x

    Yverctor.to_csv(rf'Next month returns\NextMonthReturnsForMonth-{str(month)},Year-{str(year)}.csv',index = True)


def checkingEmptyColumns(inputTable, list):
    dayNumber = 20
    firstStock = 0
    list1 = list
    i = 0
    print("checkingEmptyColumns")
    for s in list1:
        if firstStock == 0:
            if isnan(inputTable.iloc[0][32]):
                dayNumber = 19
            if isnan(inputTable.iloc[0][31]):
                dayNumber = 18
            if isnan(inputTable.iloc[0][30]):
                dayNumber = 17
            if isnan(inputTable.iloc[0][29]):
                dayNumber = 16
            firstStock = 1
        elif firstStock == 1:
            if isnan(inputTable.iloc[i][12+dayNumber]):
                print("dont have all days data")
                list1.remove(s)
                inputTable = inputTable.drop(inputTable.index[i])
        i += 1
    return list1, inputTable

def generateInputData(inputTable,month,year):
    list = symbols.copy()
    print("****************************************************************************8")
    print(month, year)
    #columns =['t+1']
    #Yverctor = pd.DataFrame(columns = columns, index=symbols)
    #####not all stocks started in 2000 so when we see stock that dont have data in the year or month we need we skip it
    badYear = False
    columNumFlag = 0
    daysNumber = 0
    index = 0
    for s in symbols:
        print("start generateInputData "+s,month,year)
        dataframe = pd.read_csv(rf'final project data 1980-2021\{s}.csv')
        dataframe['Date'] = dataframe['Date'].astype('datetime64[ns]')#change date format
        j = 13
        y = year
        for i in range(12):
            if month-j <= 0:
                if month == 1 and j == 13:
                    y = year - 2
                else:
                    y = year - 1
            else:
                y = year
            try:
                currMonth = oneMonth(dataframe,validMonth(month-j),y)
                if currMonth.empty:
                    print("badYear",validMonth(month-j),y)
                    badYear = True
                    break
            except:
                badYear = True
                break
            x = monthReturn(currMonth)
            inputTable.loc[s][i] = x
            j -= 1
        if badYear:
            badYear = False
            inputTable = inputTable.drop(inputTable.index[index])
            list.remove(s)
            continue#skip to the next stock
        #last 20 days
        nextMonth=oneMonth(dataframe,validMonth(month+1),y)
        if nextMonth.empty:
            print("badYear",validMonth(month),y)
            inputTable = inputTable.drop(inputTable.index[index])
            list.remove(s)
            continue
        j=0
        currMonth=oneMonth(dataframe,month,year)
        #nextMonth=oneMonth(dataframe,validMonth(month+1),y)
        rowCount = currMonth.shape[0]
        print(rowCount)
        if columNumFlag == 0 and daysNumber == 0:
            daysNumber = rowCount
            columNumFlag = 1
        elif columNumFlag == 1 and rowCount != daysNumber:
            if not (daysNumber >= 20 and rowCount >= 20):
                print("badYear",validMonth(month),y)
                inputTable = inputTable.drop(inputTable.index[index])
                list.remove(s)
                continue#skip to the next stock

        for i in range(12,32,1):
            if rowCount <= j or 20 <= j:
                print(j)
                break
            if j == rowCount - 1:
                x = dayReturn(currMonth.iloc[j],nextMonth.iloc[0])
            else:
                x = dayReturn(currMonth.iloc[j],currMonth.iloc[j+1])
            inputTable.loc[s][i] = x
            j+=1
        index += 1
        #creating monthly returns vector for month+1
        #y=year
        #if month+1>12:
         #   y=year+1;
        #x=monthReturn(nextMonth)
        #Yverctor.loc[s][0]=x
    #END FOR S
    if month == 1:
        inputTable.iloc[:, 32] = 1
    else:
        inputTable.iloc[:, 32] = 0
    #delete empty rows
    print(inputTable.shape[0])
    inputTable.dropna(subset = ["t-13"], inplace=True)
    print(inputTable.shape[0])#print how many stocks left
    #Yverctor.dropna(subset = ["t+1"], inplace=True)
    #Yverctor.to_csv(rf'Next month returns\NextMonthReturnsForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    #list, inputTable = checkingEmptyColumns(inputTable,list)
    inputTable.to_csv(rf'Data\inputDataForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    nextMonthReturns(month, year,list)
    print("end generateInputData")
    return inputTable, list

def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False

def CumulativeReturn(row, j):
    res = 1
    temp = 0
    if isnan(row.iloc[j+1]):
        return row.iloc[j+1]
    if j == 0:
        return row.iloc[1]
    if j == 12:
        return row.iloc[13]
    if j < 12:
        for i in range(0, j+1):
            temp = 1 + row.iloc[i+1]
            res *= temp
    else:
        for i in range(13, j):
            temp = 1 + row.iloc[i+1]
            res *= temp
    return res - 1


def updateList(inputTable):
    newList = []
    temp = inputTable.iloc[:, 0]
    for i in range(inputTable.shape[0]):
        newList.append(temp[i])
    #print(newList)
    return newList

def generateInputCumulativeReturn(month, year, list):
    s = 'inputDataForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    outputTable = createTable(list)
    #print(inputTable)
    if inputTable.shape[0] != outputTable.shape[0]:
        newList = updateList(inputTable)
        outputTable = createTable(newList)

    for i in range(inputTable.shape[0]):
        for j in range(0,32):
            outputTable.iloc[i][j] = CumulativeReturn(inputTable.iloc[i], j)
    #delete empty rows
    #print(outputTable.shape[0])
    #outputTable.dropna(subset = ["t-13"], inplace=True)
    #print(outputTable.shape[0])#print how many stocks left
    if month == 1:
        outputTable.iloc[:, 32] = 1
    else:
        outputTable.iloc[:, 32] = 0
    outputTable.to_csv(rf'Data\CumulativeReturnForMonth-{str(month)},Year-{str(year)}.csv',index = True)
    #(outputTable)
    return outputTable

def generateZscore(month,year):
    s='CumulativeReturnForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    print("in")
    print(inputTable)
    #outputTable= inputTable.copy()
    outputTable = createTable(updateList(inputTable))
    for i in range(1, outputTable.shape[1]):
        arr = stats.zscore(inputTable.iloc[:, i])
        print(arr)
        for j in range(outputTable.shape[0]):
                outputTable.iloc[j][i-1]=arr[j]
    if month == 1:
        outputTable.iloc[:, 32] = 1
    else:
        outputTable.iloc[:, 32] = 0
    outputTable.to_csv(rf'Data\ZscoreForMonth-{str(month)},Year-{str(year)}.csv', index=True)
    print("out:")
    print(outputTable)
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
'''
flagFirst = 0
for j in range(2018,2021):
    for i in range(1,13):
        if j == 1981 and flagFirst == 0:
            i = 2
            flagFirst = 1
        inputTable = createTable(symbols)
        table, list1 = generateInputData(inputTable,i,j)
        x=generateInputCumulativeReturn(i,j,list1)
        '''
flagFirst = 0
for j in range(1981,2021):
    for i in range(1,13):
        if j == 1981 and flagFirst == 0:
            i = 2
            flagFirst = 1
        generateZscore(i,j)
        #generateInputClassification(i,j)
