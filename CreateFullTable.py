import pandas as pd
import numpy as np


fullInputTable = pd.read_csv(rf'Data\inputDataForMonth-{str(1)},Year-{str(1981)}.csv',index = False)
fullClassTable = pd.read_csv(rf'Data\classForMonth-{str(1)},Year-{str(1981)}.csv',index = True)
flag = 0
for year in range(1981, 2020):
    for month in range(1, 12):
        if flag == 0:
            month = 2
            flag = 1
        pd.read_csv(rf'Data\inputDataForMonth-{str(month)},Year-{str(year)}.csv',index = False)
