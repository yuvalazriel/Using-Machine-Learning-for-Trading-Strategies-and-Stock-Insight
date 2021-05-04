import pandas as pd
import numpy as np
import preprocessing

#crate tables for each month - I replace the year every time but do it the way you like
#for m in range(1,13):
 #   inputTable=preprocessing.createTable()
  #  preprocessing.generateInputData(inputTable,m,2020)
   # x=preprocessing.generateInputCumulativeReturn(m,2020)
    #preprocessing.generateZscore(m,2020)
    #preprocessing.generateInputClassification(m,2020)


#I didnt made any changes yet - dont run them blindely
def appendMonths(month,year, endMonth,endYear):
    y = year
    s = 'ZscoreForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'Data\{s}.csv')
    ZscoreoutputTable = inputTable.copy()
    print(inputTable)
    '''
    s='ZscoreForMonth-'+str(month+1)+',Year-'+str(year)
    AppendTable = pd.read_csv(rf'Data after preprocessing\{s}.csv')
    df1=ZscoreoutputTable.append(AppendTable)
    s='ZscoreForMonth-'+str(month+1)+',Year-'+str(year)
    AppendTable = pd.read_csv(rf'Data after preprocessing\{s}.csv')
    df2=df1.append(AppendTable)
    print(df2)
    '''
    for m in range(month,13):
        s1='ZscoreForMonth-'+str(m)+',Year-'+str(year)
        AppendTable = pd.read_csv(rf'Data\{s1}.csv')
        AppendTable1=AppendTable.copy()
        #print("fpr month",m)
        #print(AppendTable1)
        df1=ZscoreoutputTable.append(AppendTable1)
        ZscoreoutputTable=df1.copy()
        #print(ZscoreoutputTable)
    m=1
    year+=1
    flag = 1
    while flag == 1:
        for m in range(1,13):
            if year == endYear and m >= endMonth:
                flag = 0
                break
            s1='ZscoreForMonth-'+str(m)+',Year-'+str(year)
            print(m , year)
            AppendTable = pd.read_csv(rf'Data\{s1}.csv')
            AppendTable1=AppendTable.copy()
            #print("fpr month",m)
            #print(AppendTable1)
            df1=ZscoreoutputTable.append(AppendTable1)
            ZscoreoutputTable=df1.copy()
            print(ZscoreoutputTable)
        year+=1

    ZscoreoutputTable.to_csv(rf'FINAL DATA\classForYear-{str(y)} to Year-{str(endYear)}.csv',index =True)


def appendMonthsVector(month,year):
    y=year
    s='classForMonth-'+str(month)+',Year-'+str(year)
    inputTable = pd.read_csv(rf'class and zscors 2017-2020\{s}.csv')
    ZscoreoutputTable= inputTable.copy()
    '''
    for m in range(2,9):
        s1='classForMonth-'+str(m)+',Year-'+str(year)
        AppendTable = pd.read_csv(rf'Data after preprocessing\{s1}.csv')
        AppendTable1=AppendTable.copy()
        #print("fpr month",m)
        #print(AppendTable1)
        df1=ZscoreoutputTable.append(AppendTable1)
        ZscoreoutputTable=df1.copy()
        #print(ZscoreoutputTable)
    
    s='ZscoreForMonth-'+str(month+1)+',Year-'+str(year)
    AppendTable = pd.read_csv(rf'Data after preprocessing\{s}.csv')
    df1=ZscoreoutputTable.append(AppendTable)
    s='ZscoreForMonth-'+str(month+1)+',Year-'+str(year)
    AppendTable = pd.read_csv(rf'Data after preprocessing\{s}.csv')
    df2=df1.append(AppendTable)
    print(df2)
    '''
    for m in range(3,13):
        s1='classForMonth-'+str(m)+',Year-'+str(year)
        AppendTable = pd.read_csv(rf'class and zscors 2017-2020\{s1}.csv')
        AppendTable1=AppendTable.copy()
        #print("fpr month",m)
        #print(AppendTable1)
        df1=ZscoreoutputTable.append(AppendTable1)
        ZscoreoutputTable=df1.copy()
        #print(ZscoreoutputTable)
    year+=1
    for m in range(1,13):
        s1='classForMonth-'+str(m)+',Year-'+str(year)
        AppendTable = pd.read_csv(rf'class and zscors 2017-2020\{s1}.csv')
        AppendTable1=AppendTable.copy()
        #print("fpr month",m)
        #print(AppendTable1)
        df1=ZscoreoutputTable.append(AppendTable1)
        ZscoreoutputTable=df1.copy()
        print(ZscoreoutputTable)
    year+=1
    for m in range(1,13):
        s1='classForMonth-'+str(m)+',Year-'+str(year)
        AppendTable = pd.read_csv(rf'class and zscors 2017-2020\{s1}.csv')
        AppendTable1=AppendTable.copy()
        #print("fpr month",m)
        #print(AppendTable1)
        df1=ZscoreoutputTable.append(AppendTable1)
        ZscoreoutputTable=df1.copy()
        print(ZscoreoutputTable)

    ZscoreoutputTable=ZscoreoutputTable.drop(ZscoreoutputTable.columns[[0]],1)
    ZscoreoutputTable.to_csv(rf'FINAL DATA\TrainVectorForYear-{str(y)} to Year-{str(year)}.csv',index =False)

appendMonths(2,1981,11,2020)