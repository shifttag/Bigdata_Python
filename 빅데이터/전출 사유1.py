import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


### 엑셀 데이터 활용
path = 'C:/bigdata/빅데이터/'  # 경로 맨 뒤에 / 붙여 놓기!  + 역 슬래쉬 -> 슬래쉬(/)로 바꾸기

#1. 엑셀 파일 로드
a = pd.read_excel(path + 'seoul20.xlsx')
print(a)

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

sns.barplot(x='시점',y='직업',data=a)
plt.ylim(0,150000)
plt.xlabel('연도')
plt.ylabel('건수')
plt.title("직업때문에 전입 수(서울)")
plt.show()

