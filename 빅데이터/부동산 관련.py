import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import datetime, timedelta

path = 'C:/bigdata/'
부동산 = pd.read_excel('부동산.xlsx', sheet_name='Sheet2')

# print(부동산)


부동산['시점'] = 부동산['시점'].astype('str')
부동산['시점'] = pd.to_datetime(부동산['시점'].apply(lambda x : x.ljust(7, '0')))
# print(부동산['시점'])

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.plot(부동산['시점'],부동산['서울'], linestyle='-', label='서울')
plt.plot(부동산['시점'],부동산['부산'], linestyle='-', label='부산')
plt.plot(부동산['시점'],부동산['경기'], linestyle='-', label='경기')

plt.ylim(80,120)

plt.title("부동산 가격")
plt.legend()
plt.show()



