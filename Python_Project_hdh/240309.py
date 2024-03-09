''' 24 - 03 - 09 강의 자료 '''
import matplotlib.pyplot as plt

''' Review 

1. 패키지 로드
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

2. 데이터 파악
mpg.head()
mpg.tail()
mpg.shape
mpg.info()  # 속성
mpg.describe()  # 요약 통계량

3. 변수명 바꾸기
mpg = mpg.rename(columns = {'cty' : 'city'})

4. 파생변수
mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2 
mpg['test'] = np.where(mpg[total] > 20, '수입', '수입x')

5. 빈도 확인
count_test = mpg['test'].value_counts()
count_test.plot.bar(rot = 0)  # rot = 0 : 변수명 기울임

'''

# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd

path = 'C:/Python_Project_hdh/Data/'
exam = pd.read_csv(path + 'exam.csv')

''' 필요한 변수 추출 '''

# a = exam['math']
# print(a)

## 여러 변수 추출 - 대괄호를 두 번 쓴다 (리스트로 만들어준다)
# a = exam[['math', 'english', 'science']]
# print(a)

# a = exam[['math']]
# print(a)

### 변수 제거

# a = exam.drop(columns='math')
# print(a)

# a = exam.drop(columns= ['math', 'english'])
# print(a)


### pandas 함수 조합

# 1. nclass 가 1 반인 행만 추출한 다음 영어 점수만 추출
# a = exam.query('nclass == 1') ['english']
# print(a)

# 2. 수학점수가 50점 이상인 학생만 추출한 다음, id, math 앞 부분 10행까지 출력
# a = exam.query('math >= 50')[['id', 'math']].head(10)
# print(a)

# a = exam.query('math >= 50')\
#     [['id', 'math']]\
#     .head(10)
# print(a)

''' 순서대로 정렬 '''

### 오름차순 정렬
# a = exam.sort_values('math')
# print(a)

### 내림차순 정렬
# a = exam.sort_values('math', ascending=False)
# print(a)

### 여러 정렬 기준 적용

# 1. 반, 수학점수 오름차순 정렬
# a = exam.sort_values(['nclass', 'math'], ascending=False)
# print(a)

''' 파생변수 추가 '''

### total 추가
# a = exam.assign(total = exam['math'] + exam['english'] + exam['science'],
#                 mean = (exam['math'] + exam['english'] + exam['science']) / 3)
# print(a)

# 1. np.where() 적용
import numpy as np
# a = exam.assign(test = np.where(exam['science'] > 60, 'pass', 'fail'))
# print(a)

# 2. 추가한 변수를 바로 pandas 함수에 활용 - total 추가, total기준 정렬
# a = exam.assign(total = exam['math'] + exam['english'] + exam['science'])\
#     .sort_values('total')
# print(a)

''' 집단별로 요약하기 '''
'''
집단별 평균이나 집단별 빈도처럼 각 집단을 요약한 값을 구할 때는 df.groupby()와
df.agg() 를 사용한다
'''

# 1. 수학 평균 구하기
# a = exam.agg(mean_math = ('math', 'mean'))
''' 요약하는데 사용할 변수명과 함수명은 따옴표로 감싸 문자 형태로 입력하고, 함수명
뒤에 괄호를 넣지 않음에 주의'''
# print(a)

# 2. 집단별 요약 통계량 구하기 ***
# a = exam.groupby('nclass')\
#     .agg(mean_math = ('math', 'mean'))
# print(a)
# print('-'*100)
# 3. 인덱스로 들어간 변수 빼내기
# a = exam.groupby('nclass', as_index=False)\
#     .agg(mean_math = ('math', 'mean'))
# print(a)

# 4. 여러 요약 통계량 한 번에 구하기
# a = exam.groupby('nclass', as_index=False)\
#     .agg(mean_math = ('math', 'mean'),          # 수학 점수 평균
#          sum_math = ('math', 'sum'),            # 수학 점수 합계
#          median_math = ('math', 'median'),      # 수학 점수 중앙값
#          n = ('nclass','count'))                # 빈도(학생 수)
# print(a)

# 5. 집단별로 다시 집단 나누기
mpg = pd.read_csv(path + 'mpg.csv')
# print(mpg)

# a = mpg.groupby(['manufacturer', 'drv'])\
#     .agg(mean_cty = ('cty', 'mean'))
# print(a)

# 6. pandas 함수 조합
'''
Q. 제조 회사별로 'suv' 자동차의 도시 및 고속도로 합산 연비 평균을 구해 내림차순으로 정렬하고 1~5등 출력
'''

# a = mpg.query('category == "suv"') \
#     .assign(total = ((mpg['cty'] + mpg['hwy']) / 2)) \
#     .groupby('manufacturer') \
#     .agg(mean_total = ('total', 'mean')) \
#     .sort_values('mean_total', ascending=False) \
#     .head(5)
# print(a)

''' 데이터 합치기 '''

### 가로로 합치기

# 1. 데이터 생성
test1 = pd.DataFrame({'id':[1,2,3,4,5],
                      'mid':[60,80,70,90,85]})
test2 = pd.DataFrame({'id':[1,2,3,4,5],
                      'final':[70,93,65,98,80]})
# print(test1);print(test2)

# 2. id 기준으로 합쳐서 total에 할당
total = pd.merge(test1, test2, how='left', on='id')
# print(total)

# 3. 다른 데이터를 활용해서 변수 추가

''' 여기 못적음 '''

### 세로로 합치기
g_a = pd.DataFrame({'id':[1,2,3,4,5],
                      'mid':[60,80,70,90,100]})
g_b = pd.DataFrame({'id':[6,7,8,9,10],
                      'final':[70,93,65,98,80]})
# print(g_a)
# print(g_b)

g_all = pd.concat([g_a, g_b])
# print(g_all)

# ----------------------------------------------------------------------------------------------------------------------
''' 데이터 정제 - 빠진 데이터, 이상한 데이터 제거 '''

''' 결측치 : 누락된 값, 비어있는 값 '''
''' 실제 데이터 분석할 때는 결측치가 있으면 빼고 분석을 해야 올바른 분석 '''

### 결측치 찾기

# 1. 결측치 임의 생성 - numpy 패키지의 np.nan을 입력하면 결측치 생성
import numpy as np
import pandas as pd

df = pd.DataFrame({'성별' : ['M','F',np.nan, 'M','F'],
                   'score' : [5,4,3,4,np.nan]})
# print(df)
''' NaN 값이 있는 상태로 연산을 하면 출력 결과도 NaN이 된다. '''

# 2. 결측치 확인하기 - pd.isna()
# print(pd.isna(df))

# 3. 결측치 빈도 확인
# print(pd.isna(df).sum())

# 4. 결측치 제거 - pd.dropna()
# a = df.dropna(subset=['score']) # score 변수에 결측치 제거
# print(a)

# 5. 여러 변수에 결측치 없는 데이터 추출
# a = df.dropna(subset=['score', '성별'])
# print(a)

# 6. 결측치가 하나라도 있으면 제거
''' 이 방법은 간편하긴 하지만 분석에 필요한 행까지 손실된다는 단점이 있다. '''
# df_nomiss = df.dropna()
# print(df_nomiss)

### 결측치 대체 - 데이터의 크기가 작고 결측치가 많은 경우에 사용
'''
방법 1. 평균이나 최빈값 같은 대표값을 구해서 결측치를 하나의 값으로 일괄 대체
방법 2. 통계 분석 기법으로 예측값을 추정해 대체하는 방법
'''
# 1. 평균으로 결측치 대체
exam = pd.read_csv(path + 'exam.csv')
exam.loc[[2,7,14], ['math']] = np.nan   # 결측치 임의 생성
# print(exam)

# print(exam['math'].mean())  # 대체할 평균값 출력 = 55

# 2. NaN값 55로 대체
exam['math'] = exam['math'].fillna(55)
# print(exam)
# print(exam['math'].isna().sum())    # 대체 확인
# print(exam['math'].mean())

''' 이상치 : 정상 범위를 크게 벗어난 값  '''

### 이상치 제거 - 존재할 수 없는 값

df = pd.DataFrame({'성별' : [1,2,1,3,2,1],
                   'score' : [5,4,3,4,2,6]})

# 1. 이상치 확인
a = df['성별'].value_counts().sort_index()
''' 데이터.value_counts() 에 sort_index()를 하면 빈도 기준으로 내림차순 정렬 하지 않고
    변수의 값 순서로 정렬 한다. '''
# print(a)

a = df['score'].value_counts().sort_index()
# print(a)

# 2. 결측 처리하기 - 성별이 3이면 NaN 부여
df['성별'] = np.where(df['성별'] == 3, np.nan, df['성별'])
# print(df)

df['score'] = np.where(df['score'] > 5 , np.nan, df['score'])
# print(df)

# 3. 결측치를 제거하고 분석
a = df.dropna()\
    .groupby('성별')\
    .agg(mean_score = ('score', 'mean'))
# print(a)

### 이상치 제거 - 극단적인 값 -> 극단치 : 논리적으로는 존재할 수 있지만 너무 극단적으로 크거나 작은 값
'''
극단치를 제거하려면 먼저 어디까지가 정상 범위인지 정해야한다.

1. 논리적으로 판단
2. 통계적인 기준을 이용
'''

# 1. 상자 그림으로 극단치 기준을 정하기
# print(mpg)

import seaborn as sns
# sns.boxplot(data=mpg, y = 'hwy')
# plt.show()

# 2. 극단치 기준값 구하기

# 2-1) 1사분위수, 3사분위수 구하기
# pct25 = mpg['hwy'].quantile(.25)    # 분위수 함수
# print('1사분위수 :',pct25)
# pct75 = mpg['hwy'].quantile(.75)    # 분위수 함수
# print('3사분위수 :',pct75)

# 2-2) IQR 구하기
# iqr = pct75 - pct25
# print('IQR :', iqr)

# 2-3) 하한, 상한 구하기
'''
극단치의 경계가 되는 하한, 상한
    하한 : 1사분위수보다 IQR의 1.5배 만큼 더 작은 값
    상한 : 3사분위수보다 IQR의 1.5배 만큼 더 큰 값
'''
# print('하한 :', pct25 - iqr * 1.5)
# print('상한 :', pct75 + iqr * 1.5)

''' 고속도로 연비가 4.5보다 작거나 40.5보다 크면 상자 그림 기준으로 극단치다. '''

# 2-4) 극단치를 결측 처리
# mpg['hwy'] = np.where((mpg['hwy'] < 4.5) | (mpg['hwy'] > 40.5), np.nan, mpg['hwy'])

# 2-5) 결측치 빈도 확인
# result = mpg['hwy'].isna().sum()
# print(result)

# 2-6) 결측치 제거하고 분석
# a = mpg.dropna(subset=['hwy'])\
#     .groupby('drv')\
#     .agg(mean_hwy = ('hwy', 'mean'))
# print(a)

# ----------------------------------------------------------------------------------------------------------------------
''' 파이썬 그래프 '''

'''
산점도 : 데이터를 x축과 y축에 점으로 표현한 그래프

나이와 소득처럼 연속된 값으로 된 두 변수의 관계를 표현할 때
'''

import seaborn as sns
import pandas as pd

mpg = pd.read_csv(path + 'mpg.csv')

### displ(배기량) 과 hwy(고속도로 연비) 관계 산점도
# sns.scatterplot(data=mpg, x = 'displ', y = 'hwy')
# plt.show()

# 1. x축과 y축 범위 제한 3~6
# sns.scatterplot(data=mpg, x = 'displ', y = 'hwy')\
#     .set(xlim = (3,6), ylim = (10,30))
# plt.show()

# 2. 종류별로 표식 색깔 바구기
# sns.scatterplot(data=mpg, x = 'displ', y = 'hwy', hue='drv')
# plt.show()

''' 그래프 설정 바꾸기 '''
# plt.rcParams.update({'figure.dpi' : '150',              # 해상도, 기본값 72
#                      'figure.figsize' : [8,6],          # 가로, 세로 크기 (기본값 [6,4])
#                      'figure.size' : '15',              # 글자 크기 (기본값 10)
#                      'font.family' : 'Malgun Gothic'})  # 폰트

''' 
막대 그래프 : 데이터의 크기를 막대의 길이로 표현한 그래프
 -> 집단 간의 차이를 표현할 때 자주 사용
'''

### 평균 막대 그래프

# 1. 집단별 평균표 생성
df_mpg = mpg.groupby('drv', as_index=False)\
    .agg(mean_hwy = ('hwy', 'mean'))
# print(df_mpg)

# 2. 그래프 생성
# sns.barplot(data=df_mpg, x= 'drv', y='mean_hwy')
# plt.show()

# 3. 크기순으로 정렬
''' 막대의 정렬 순서는 그래프를 만드는데 사용한 데이터 프레임의 행 순서에 따라 정렬된다. '''
# df_mpg = df_mpg.sort_values('mean_hwy', ascending=False)
# sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
# plt.show()

### 빈도 막대 그래프

# 1. 집단별 빈도표 만들기
df_mpg = mpg.groupby('drv',as_index=False)\
    .agg(n = ('drv', 'count'))
# print(df_mpg)

# 2. 그래프 생성
# sns.barplot(data=df_mpg, x = 'drv', y = 'n')
# plt.show()

# 3. 빈도표 없이 빈도 그래프 생성할 때는 countplot을 쓴다.
# sns.countplot(data=mpg, x = 'drv')
# plt.show()

# 4. countplot 막대 정렬
# sns.countplot(data=mpg, x='drv', order=['4','f','r'])
# plt.show()

# 4-1 drv의 값을 빈도가 높은 순서로 정렬
# a = mpg['drv'].value_counts().index
# print(a)

# 4-2 drv 빈도 높은 순으로 막대 정렬
# sns.countplot(data=mpg, x= 'drv',
#               order=mpg['drv'].value_counts().index)
# plt.show()

''' 
선 그래프 : 데이터를 선으로 표현한 그래프
 -> 시간에 따라 달라지는 데이터를 표현 할 때 자주 사용
 -> 일정 시간 간격을 두고 나열된 데이터 : 시계열 데이터
 '''

# economics 데이터 로드
economics = pd.read_csv(path + 'economics.csv')
# print(economics.head())

# sns.lineplot(data=economics, x='date', y='unemploy')
# plt.show()

# 1. x축에 연도 표시하기

# 1-1 날짜 시간 타입(datetime64) 변수 생성
# print(economics.info())
economics['date2'] = pd.to_datetime(economics['date'])
# print(economics.info())

# 1-2 연 추출, 월 추출, 일 추출
# print(economics[['date', 'date2']])

# print(economics['date2'].dt.year)         # 연 추출
# print(economics['date2'].dt.month)        # 월 추출
# print(economics['date2'].dt.day)          # 일 추출

# 1-3 연도 변수 생성하고 x축에 연도 표시
economics['year'] = economics['date2'].dt.year
# print(economics,head())
# sns.lineplot(data=economics, x = 'year', y='unemploy', ci = None)   # ci = None : 신뢰구간(선그래프에 반투명 부분) 표시 x
# plt.show()

''' 상자 그림 : 데이터 분포 또는 퍼져있는 형태를 직사각형 상자 모양으로 표현한 그래프 '''
# sns.boxplot(data=mpg, x='drv', y='hwy')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
'''
한국복지패널데이터
: 전국에서 7천여 가구를 선정해서 2006년도 부터 매년 추적 조사한 자료. 천여개의 변수로 구성된 데이터
'''
### 패키지 설치
''' 사용할 데이터 파일이 통계 분석 소프트웨어인 SPSS 전용 파일이다. pyreadstat 패키지를 설치를 해준다

Terminal -> Command Prompt -> pip install pyreadstat
'''
import pandas as pd
import numpy as np
import seaborn as sns

### 데이터 로드
raw_welfare = pd.read_spss(path + 'Koweps_hpwc14_2019_beta2.sav')

# 복사본 생성
welfare = raw_welfare.copy()

### 데이터 검토
# print(welfare.info())
# print(welfare.describe())

### 변수명 수정
welfare = welfare.rename(
    columns={'h14_g3' : 's',                # 성별
             'h14_g4' : 'birth',            # 태어난 년도
             'h14_g10' : 'marriage_type',   # 혼인 여부
             'h14_g11' : 'religion',        # 종교
             'p1402_8aq1' : 'income',       # 월급
             'h14_eco9' : 'code_job',       # 직업 코드
             'h14_reg7' : 'code_region'}    # 지역 코드
)
'''
- 데이터 분석 절차
1. 사용할 변수 검토 및 전처리 - 이상치와 결측치 정제
2. 변수 간 관계 분석 - 요약표나 그래프 생성하고 해석
'''

''' 성별에 따른 월급 차이 '''

### 성별 변수 검토
# print(welfare['s'].dtypes)    # float64 형식
# print(welfare['s'].value_counts())  # -> 이상치 확인도 가능 -> 이상치 없음을 확인 # 여성 : 7913, 남성 : 6505

# 성별 항목에 이름을 부여
welfare['s'] = np.where(welfare['s'] == 1, 'Male', 'Female')

# sns.countplot(data=welfare, x='s')
# plt.show()

### 월급 변수 검토
# print(welfare['income'].dtypes)
# print(welfare['income'].describe())
# sns.histplot(data=welfare, x='income')
# plt.show()

# 전처리
a = welfare['income'].isna().sum()
# print(a)    # 결측치가 9884개가 존재한다. -> 직업이 없어서 월급을 받지 않는 응답자기 있기 때문

# 이상치 결측 처리
welfare['income'] = np.where(welfare['income'] == 9999, np.nan, welfare['income'])
# print(welfare['income'].isna().sum()) # 무응답인 9999는 없었다 라는 결론

### 성별에 따른 월급 차이 - 성별에 따라 월급 차이가 있을까?

# 1. 성별 월급 표 생성
# s_income = welfare.dropna(subset=['income'])\
#     .groupby('s', as_index=False)\
#     .agg(mean_income = ('income', 'mean'))
# print(s_income)

# 2. 시각화
# sns.barplot(data=s_income, x='s', y='mean_income')
# plt.show()

''' 나이와 월급의 관계 - 몇 살때 월급을 가장 많이 받을까? '''

### 나이 변수 검토
# print(welfare['birth'].dtypes)
# print(welfare['birth'].describe())

# sns.histplot(data=welfare, x='birth')
# plt.show()

# 전처리
# print(welfare['birth'].isna().sum())

welfare['birth'] = np.where(welfare['birth'] == 9999, np.nan, welfare['birth'])
# print(welfare['birth'].isna().sum())  # 이상치와 결측치 둘다 없음을 확인할수 있음

# 파생변수 생성 - 나이 변수 생성
welfare = welfare.assign(age = 2019 - welfare['birth'] + 1)
# print(welfare['age'].describe())

### 나이와 월급의 관계 분석
age_income = welfare.dropna(subset=['income'])\
    .groupby('age')\
    .agg(mean_income = ('income', 'mean'))
# print(age_income)

### 시각화
# sns.lineplot(data=age_income, x='age', y='mean_income')
# plt.show()

''' 종교 유무에 따른 이혼율 - 종교가 있으면 이혼을 덜 할까? '''

### 종교 변수 검토
# print(welfare['religion'].dtypes)
# print(welfare['religion'].value_counts())     # 이상치 없음, 결측치 없음

# 종교 유무 이름 부여
welfare['religion'] = np.where(welfare['religion'] == 1, 'Yes', 'No')
# print(welfare['religion'].value_counts())
sns.countplot(data=welfare, x='religion')
# plt.show()

### 혼인상태 변수 검토 및 전처리
# print(welfare['marriage_type'].dtypes)
# print(welfare['marriage_type'].value_counts())

# 파생변수 - 이혼 여부 생성
welfare['marriage'] = np.where(welfare['marriage_type'] == 1 , 'marriage',
                               np.where(welfare['marriage_type'] == 3, 'divorce', 'etc'))

# 이혼 여부별 빈도
n_divorce = welfare.groupby('marriage', as_index=False)\
    .agg(n=('marriage', 'count'))
# print(n_divorce)

### 종교 유무에 따른 이혼율 분석

# 1. 종교 유무에 따른 이혼율 표 생성
rel_div = welfare.query('marriage != "etc"')\
    .groupby('religion', as_index=False)\
    ['marriage']\
    .value_counts(normalize=True)   # 비율 구하는 옵션
# print(rel_div)

# 2. 시각화
''' 이혼에 해당하는 값만 추출한 다음 proportion를 백분율로 바꾸고 소수점 첫째자리까지 반올림 하겠다. '''
rel_div = rel_div.query('marriage == "divorce"')\
    .assign(proportion = rel_div['proportion'] * 100)\
    .round(1)
# print(rel_div)

sns.barplot(data=rel_div, x='religion', y='proportion')
plt.show()
''' 출력한 표와 그래프를 보면, 이혼율을 종교가 있으면 8.0%, 종교가 없으면 9.5%다.
    따라서 종교가 있는 사람이 이혼을 덜 한다고 볼 수 있다.'''


