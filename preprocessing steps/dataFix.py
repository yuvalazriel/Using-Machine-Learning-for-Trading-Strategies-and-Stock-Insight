import pandas as pd


def isnan(value):
    try:
        import math
        return math.isnan(float(value))
    except:
        return False


def checkNan(month, year):
    outputTable = pd.read_csv(rf'FinalData\ZscoreForMonth-{str(month)},Year-{str(year)}.csv')
    rowCount = outputTable.shape[0]
    colCount = outputTable.shape[1]
    flag = 0
    jan = 0
    if month == 12:
        jan = 1
    for i in range(rowCount):
        for j in range(colCount):
            if j == 32 and jan == 1:
                outputTable.iloc[i][j] = 1
                flag = 1
            if pd.isnull(outputTable.iloc[i][j]):
                outputTable.iloc[i][j] = 0
                flag = 1
    if flag == 1:
        outputTable.to_csv(rf'FinalData\ZscoreForMonth-{str(month)},Year-{str(year)}.csv',index = False)
        print(str(month) + str(year))


for y in range(2017,2021):
    for m in range(1,13):
        if y == 2017 and m == 1:
            continue
        if y == 2020 and ( m == 9 or m == 10 or m == 11 or m == 12 ):
            continue
        checkNan(m, y)

