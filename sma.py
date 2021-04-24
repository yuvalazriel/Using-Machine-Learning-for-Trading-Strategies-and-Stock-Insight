



def sma5(fileName):
    with open('Data\{fileName}') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < 60:
                line_count += 1
            else:
                symbols.append(row[0])
                line_count += 1





def sma20():







def sma60():



