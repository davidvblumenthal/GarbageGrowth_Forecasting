import requests
from lxml import html 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np 
import csv
from datetime import date, timedelta

def webscraper(url):

    Dates = []
    Names = []

    for i in url:
        request = requests.get(i)
        html_code = request.content

        tree = html.fromstring(html_code)

        columnDate = tree.xpath("//*[@id=\"page-content\"]/main/article/section[1]/div/table/thead/tr/th[1]/text()")
        columnName = tree.xpath("//*[@id=\"page-content\"]/main/article/section[1]/div/table/thead/tr/th[1]/text()")

        columnDateValue = tree.xpath("//*[@id=\"page-content\"]/main/article/section/div/table/tbody/tr/td[1]/text()")
        columnNameValue = tree.xpath("//*[@id=\"page-content\"]/main/article/section/div/table/tbody/tr/td[2]/a/text()")
        # 0 bis 12 sind die arbeitsfreien Feiertage 
        Dates.extend(columnDateValue[0:12])
        Names.extend(columnNameValue[0:12])
    

    StringDates = [str(s) for s in Dates]

    
    dr = pd.date_range(start='2020-01-01', end='2021-12-31')
    df = pd.DataFrame()
    df['Date'] = dr
    holidays = StringDates = [str(s) for s in Dates]
    df['Holiday'] = (df['Date'].isin(holidays)).astype(int)

    print(df)    
    df.to_csv(r'..\data\raw\holidays.csv', index = False)