from bs4 import BeautifulSoup
from requests_html import HTML, HTMLSession
from selenium import webdriver
PATH='C:\Program Files (x86)\chromedriver.exe'
driver= webdriver.Chrome(PATH)

driver.get("https://www.tipranks.com/stock-ratings?llf=sub-header-stock-ratings-page-link")
print(driver.title)
#client-pages-ScreenerRatingsPage-ScreenerRatingsPage__previewTableContainer
#client-components-ReactTableWrapper-cells__StockNameCell
# client-components-ReactTableWrapper-cells__StockNameCell
#stocksNames=driver.find_elements_by_class_name("client-components-ReactTableWrapper-cells__nameCellCompanyName")
#print(stocksNames)
table=driver.find_element_by_class_name("client-pages-ScreenerRatingsPage-ScreenerRatingsPage__previewTableContainer")
stocksNames=table.find_elements_by_class_name("client-components-ReactTableWrapper-cells__nameCellCompanyName")
stokSymbols= table.find_elements_by_class_name("client-components-ReactTableWrapper-cells__StockNameCell")

for i in range(len(stocksNames)):
    print(stocksNames[i].text)




driver.quit()
