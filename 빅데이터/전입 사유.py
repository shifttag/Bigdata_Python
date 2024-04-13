import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

### 엑셀 데이터 활용
path = 'C:/bigdata/빅데이터/'  # 경로 맨 뒤에 / 붙여 놓기!  + 역 슬래쉬 -> 슬래쉬(/)로 바꾸기

#1. 엑셀 파일 로드
a = pd.read_excel(path + 'seoul20_1.xlsx')
# print(a)


a['total'] = (a['2014년']+a['2015년']+a['2016년']+a['2017년']+a['2018년']+a['2019년']+a['2020년']+a['2021년']+a['2022년']+a['2023년'])/10
# print(a.head())

total_mean = a['total']
print(total_mean)
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

plt.title('서울로 전입한 사람들의 전입 사유')
sns.barplot(x ='전입사유별', y='total',data=a)



plt.xlabel('전입 사유')
plt.ylabel('건수')
plt.show()

