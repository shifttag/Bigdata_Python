import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

path = 'C:/bigdata/'
일자리 = pd.read_excel('일자리.xlsx')
print(일자리)

일자리['서울+경기'] = 일자리['서울 취업자 (천명)'] + 일자리['경기 취업자 (천명)']

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['font.size'] = 10

plt.xlabel('년도')
plt.ylabel('(천명)')

plt.plot(일자리['시점'],일자리['서울+경기'], linestyle='-', marker='o', label='수도권')
plt.plot(일자리['시점'],일자리['부산 취업자 (천명)'], linestyle='-', marker='s', label='부산')



for i, v in enumerate(일자리['시점']):
     cc = "{:,g}".format(일자리['서울+경기'][i])
     print(v,일자리['서울+경기'][i],일자리['서울+경기'][i])
     plt.text(v,일자리['서울+경기'][i],cc,fontsize=8,color='#000000',
               horizontalalignment='center',verticalalignment='bottom')

for i, v in enumerate(일자리['시점']):
     cc = "{:,g}".format(일자리['부산 취업자 (천명)'][i])
     print(v,일자리['부산 취업자 (천명)'][i],일자리['부산 취업자 (천명)'][i])
     plt.text(v,일자리['부산 취업자 (천명)'][i],cc,fontsize=8,color='#000000',
               horizontalalignment='center',verticalalignment='bottom')

plt.title("일자리 수")
plt.legend()
plt.show()