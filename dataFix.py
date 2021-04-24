import pandas as pd
import numpy as np
import argparse
from datetime import date
import tzlocal
import csv
import preprocessing as p
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()
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

print(symbols)

for s in symbols:
    args = parse_args()
    datapath = (rf'FirstData\{s}.csv')
    #Simulate the header row isn't there if noheaders requested
    skiprows = 1 if args.noheaders else 0
    header = None if args.noheaders else 0
    dataframe = pd.read_csv(datapath)
    if not args.noprint:
        clist = ['t','o','h','l','c','v','s']
        df=dataframe.copy()
        df=df[clist]

    datesArr = pd.Series(df['t'].values.astype(float))
    local_timezone = tzlocal.get_localzone() # get pytz timezone
    i=0
    for d in datesArr:
        local_time = date.fromtimestamp(d)#, local_timezone).strftime('%d-%m-%Y')
        df['t'].iloc[i]=pd.to_datetime(local_time)
        df['s'].iloc[i]=0
        i+=1
    df.rename(columns={'t': 'Date', 'o': 'Open','h':'High','l':'Low','c':'Close','v':'Volume','s':'openinterest'},inplace=True)
    df.set_index('Date', inplace = True)
    df.to_csv(rf'FirstData\{s}.csv',index = True)



symbols = []
with open('constituents_csv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            symbols.append(row[0])
            line_count += 1


p.symbolsSet()
for y in range(2017, 2021):
    if y == 2017:
        for m in range(2, 13):
            df = p.createTable()
            p.generateInputData(df, m, y)
            outputTable = p.generateInputCumulativeReturn(m, y)
            outputTable = p.generateZscore(m, y)
            p.generateInputClassification(m, y)
    elif y == 2020:
        for m in range(1, 9):
            df = p.createTable()
            p.generateInputData(df, m, y)
            outputTable = p.generateInputCumulativeReturn(m, y)
            outputTable = p.generateZscore(m, y)
            p.generateInputClassification(m, y)
    elif y == 2018 or y == 2019:
        for m in range(1, 13):
            df = p.createTable()
            p.generateInputData(df, m, y)
            outputTable = p.generateInputCumulativeReturn(m, y)
            outputTable = p.generateZscore(m, y)
            p.generateInputClassification(m, y)



sym = []
with open('constituents_csv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            sym.append(row[0])
            line_count += 1
p.symbolsSet(sym)

for y in range(2017, 2021):

    if y == 2017:
        for m in range(2, 13):
            df = p.createTable()
            p.generateInputData(df, m, y)
            p.generateInputCumulativeReturn(m, y)
            p.generateZscore(m, y)
            p.generateInputClassification(m, y)
  
    if y == 2020:
        for m in range(1, 9):
            df = p.createTable()
            p.generateInputData(df, m, y)
            p.generateInputCumulativeReturn(m, y)
            p.generateZscore(m, y)
            p.generateInputClassification(m, y)
    if y == 2018 or y == 2019:
        for m in range(1, 13):
            df = p.createTable()
            p.generateInputData(df, m, y)
            p.generateInputCumulativeReturn(m, y)
            p.generateZscore(m, y)
            p.generateInputClassification(m, y)

'''
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

'''


for year in range(2017, 2020):
    month = 12
    outputTable = pd.read_csv(rf'FinalData\ZscoreForMonth-{str(month)},Year-{str(year)}.csv')
    rowCount = outputTable.shape[0]
    colCount = outputTable.shape[1]
    flag = 0
    for i in range(1,rowCount):
        outputTable.iloc[i][colCount-1] = 1
    outputTable.to_csv(rf'FinalData\ZscoreForMonth-{str(month)},Year-{str(year)}.csv',index = True)
'''