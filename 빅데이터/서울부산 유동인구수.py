import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib



path = 'C:/bigdata/'
유동인구 = pd.read_excel('유동인구.xlsx')

# print(유동인구)

유동인구['전입자-전출자(서울)'] = (유동인구['총전입(서울)']) - (유동인구['총전출(서울)'])

df1 = 유동인구['전입자-전출자(서울)']
# print(유동인구['전입자-전출자(서울)'])
유동인구['전입자-전출자(경기)'] = (유동인구['총전입(경기)']) - (유동인구['총전출(경기)'])

유동인구['전입자-전출자(부산)'] = (유동인구['총전입(부산)']) - (유동인구['총전출(부산)'])
# print(유동인구['전입자-전출자(부산)'])


print(유동인구)

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False


plt.plot(유동인구['시점'],유동인구['전입자-전출자(서울)'], linestyle='-', marker='o', label='서울')
plt.plot(유동인구['시점'],유동인구['전입자-전출자(부산)'], linestyle='-', marker='o', label='부산')
plt.plot(유동인구['시점'],유동인구['전입자-전출자(경기)'], linestyle='-', marker='o', label='경기')

for i, v in enumerate(유동인구['시점']):
     cc = "{:,g}".format(유동인구['전입자-전출자(서울)'][i])
     print(v,유동인구['전입자-전출자(서울)'][i],유동인구['전입자-전출자(서울)'][i])
     plt.text(v,유동인구['전입자-전출자(서울)'][i],cc,fontsize=9,color='#000000',
               horizontalalignment='center',verticalalignment='bottom')

for i, v in enumerate(유동인구['시점']):
     cc = "{:,g}".format(유동인구['전입자-전출자(부산)'][i])
     print(v,유동인구['전입자-전출자(부산)'][i],유동인구['전입자-전출자(부산)'][i])
     plt.text(v,유동인구['전입자-전출자(부산)'][i],cc,fontsize=9,color='#000000',
               horizontalalignment='center',verticalalignment='bottom')

for i, v in enumerate(유동인구['시점']):
     cc = "{:,g}".format(유동인구['전입자-전출자(경기)'][i])
     print(v,유동인구['전입자-전출자(경기)'][i],유동인구['전입자-전출자(경기)'][i])
     plt.text(v,유동인구['전입자-전출자(경기)'][i],cc,fontsize=9,color='#000000',
               horizontalalignment='center',verticalalignment='bottom')

plt.title("유동인구(전입자-전출자)")
plt.legend()
plt.show()

