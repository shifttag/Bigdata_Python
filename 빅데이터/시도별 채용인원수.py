import pandas as pd
import seaborn as sns # 그래프 패키지
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 맑은고딕 폰트 사용


### 엑셀 데이터 활용
path = 'C:/bigdata/빅데이터/'  # 경로 맨 뒤에 / 붙여 놓기!  + 역 슬래쉬 -> 슬래쉬(/)로 바꾸기

#1. 엑셀 파일 로드
a = pd.read_excel(path + '시도별.xlsx')
print(a)

# 시도별 'total' 파생변수 만들기
a['채용인원'] = (a['a2020.1'] + a['a2020.2'] + a['a2021.2'] + a['a2021.2'] + a['a2022.1'] + a['a2022.2'] +a['a2023.1'] +a['a2023.2'])/8
print(a)

# total 값을 비교하는 시각화 그래프 만들기
sns.barplot(x ='시도별', y='채용인원',data=a)
plt.title('2020년 ~ 2023년 시도별 채용인원 수')
plt.show()
