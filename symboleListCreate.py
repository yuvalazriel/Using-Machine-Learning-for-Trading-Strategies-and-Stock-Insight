import pandas as pd

def listSymbolsCreate():
    symbols = []
    inputTable = pd.read_csv(rf'sp500_history.csv')
    for i in range(881):
        flag = 0
        s = inputTable.get_value(i,"value")
        '''
        if s == "RRC" or s == "LIFE" or s == "RX" or s =="AYE" or s =="WFR" or s =="NYX" or s =="JNY"or s =="MI" or s=="TLAB" \
                or s=="MOLX" or s=="KSE" or s=="SLE" or s=="MFE" or s=="PCL" or s=="GLK" or s=="PTV" or s=="MEE" \
                or s=="RSH" or s=="ABK" or s== "SLR" or s=="TRB" or s=="CBE" or s=="NSM" or s=="CEG" or s=="SVU" \
                or s=="EP" or s=="GR" or s=="HNZ" or s=="SAI" or s=="CSRA" or s=="TWX" or s=="ANDV" or s=="CA" \
                or s=="AET" or s=="ESRX" or s=="ABS" or s=="SGP" or s=="CVG" or s=="NFX" or s=="PETM" or s=="TIE" \
                or s=="CVH" or s=="BMS" or s=="PCP" or s=="EVHC" or s=="LKQ" or s=="DLR" or s=="CCE" :
            continue

            '''
        try:
            temp = pd.read_csv(rf'final project data 1980-2021\{s}.csv')
        except:
            flag = 1
        if flag == 0:
            if s in symbols:
                continue
            symbols.append(s)
    return symbols
