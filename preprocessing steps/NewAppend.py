import pandas as pd
import numpy as np


def validMonth(val):
    if val == 0 or val == 12 or val == -12:
        return 12
    return val % 12


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False

def nanFix(df):
    #print(df.iloc[0][32])
    if isnan(df.iloc[0][32]):
        df.iloc[:, 32] = 0
    return df


def appendMonths(month, year, endMonth, endYear):
    y = year
    m = month

    s = 'ZscoreForMonth-'+str(month)+',Year-'+str(year)
    inputTable1 = pd.read_csv(rf'Data\{s}.csv')
    dfZscore = inputTable1.copy()
    dfZscore = nanFix(dfZscore)

    s = 'classForMonth-'+str(month)+',Year-'+str(year)
    inputTable2 = pd.read_csv(rf'Data\{s}.csv')
    dfClass = inputTable2.copy()

    flag = True
    while flag:
        m = validMonth(m + 1)
        if m == 1:
            y += 1
        if m == endMonth and y == endYear:
            flag = False
        s = 'ZscoreForMonth-'+str(m)+',Year-'+str(y)
        inputTable1 = pd.read_csv(rf'Data\{s}.csv').copy()
        inputTable1 = nanFix(inputTable1)
        dfZscore = dfZscore.append(inputTable1)

        s = 'classForMonth-'+str(m)+',Year-'+str(y)
        inputTable2 = pd.read_csv(rf'Data\{s}.csv').copy()
        dfClass = dfClass.append(inputTable2)

    dfClass.to_csv(rf'FINAL DATA\classForYear-{str(year)} to Year-{str(endYear)}.csv', index=True)
    dfZscore.to_csv(rf'FINAL DATA\vectorForYear-{str(year)} to Year-{str(endYear)}.csv', index=True)

appendMonths(2,1981,11,2020)
