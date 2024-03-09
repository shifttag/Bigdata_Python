''' 24 - 03 - 02 강의 자료 '''

# print("안녕하세요")
# print("Hello World!")   # print예제

'''
실행 단축키 : ctrl + shift + F10
한 줄 주석 단축키 : ctrl + /
되돌리기 : ctrl + z
코드 검색 : ctrl + f
검색 바꾸기 : ctrl + r
'''

'''
주석 : 코드 실행 결과에 영향을 미치지 않는 텍스트, 주로 코드 설명이나 하고싶은 말을 넣을 때 사용한다

1. 코드 맨 앞에 #을 붙혀준다 -> 한 줄 주석 ( 단축키 : ctrl + / )
2. ''' ''' 안에 텍스트를 써준다. -> 장문 주석
 '''

# ----------------------------------------------------------------------------------------------------------------------
''' 
변수 : 변하는 수 
1. 변수는 데이터 분석의 대상
2. 상수(하나의 값으로만 되어있는 수)는 분석할 게 없다

소득  성별  학점  국적
1천   남    3.8  대한민국
2만   남    4.2  대한민국
3만   여    2.6  대한민국
4만   여    4.5  대한민국
'''

### 변수 만들기

a = 1 # a 에 1을 할당
# print(a) # a를 출력

b = 2
c = 3
d = 4.5

# print(a+b)
# print(5*d)

'''
- 변수명 규칙
a_b_c_d_e = 15 '스네이크 법칙을 따른다'

-> 변수명은 문자, 숫자, 언더바(_)를 조합해서 정할 수 있다.
-> 반드시 문자로 시작해야 한다.
-> 한글 변수명 가능하나, 되도록 영문 변수명을 사용하자.
-> 대소문자 구분을 한다. 소문자로 만드는 습관을 권장
'''

# 여러 값으로 구성된 변수
var1 = [1,2,3] # []에 쉼표로 값을 나열해 만든 자료 구조를 리스트(list)라 한다
# print(var1)
# print(type(var1))
var2 = [4,5,6]
# print(var1 + var2) # 리스트의 더하기 연산은 합치기다.

### 문자로 된 변수
''' 파이썬에서 문자열은 큰 따옴표나 작은 따옴표로 표현할 수 있다, 여는 따옴표랑 닫는 따옴표가 같아야 한다'''
str1 = 'a'
str2 = "a"

str3 = 'text'
str4 = 'Hello World!'

str5 = ['apple', 'banana', 'orange']
str6 = ['15', '20', '45'] # 따옴표로 감싸진 숫자는 숫자의 기능을 하지 않는다. ex) '2024'

# print(str2 + str3) # 문자열의 더하기 연산은 문자열을 이어준다

# print(str3 + '/' + str4)

### 문자로 된 변수로는 연산을 할 수 없다
# str3 + 3

''' 함수 : 기능을 하는 수 
-> 입력을 받아서 출력을 내는 기능
-> 
        오렌지(입력)     믹서기(함수)     오렌지 주스(출력)
'''

### 함수 활용
x = [1,2,3]

sum(x)  # 총 합
# print(sum(x))
# print(max(x))

### 함수의 결과물로 새 변수 만들어 사용한다
x_sum = sum(x)
# print(x_sum)

''' print() : 출력 함수 '''

''' 패키지(Package) : 변수와 함수의 꾸러미 , 패키지 안에는 다양한 변수나 함수가 존재한다. '''

### 패키지 활용 : 패키지 설치 -> 패키지 로드 -> 함수 사용

''' 파이썬에서 패키지를 로드할 때 사용하는 구문 : import 패키지명 '''
import seaborn  # 그래프를 만들 때 자주 사용하는 패키지
import matplotlib.pyplot as plt
import matplotlib

### 패키지 함수 사용
var = ['a', 'a', 'b', 'c']
# seaborn.countplot(x = var)
# plt.show()  # 그래프 출력 함수

### 패키지 약어 사용하기
import seaborn as sns # seaborn 패키지는 통상적으로 sns라 줄여쓴다
# sns.countplot(x = var)
# plt.show()

### seaborn 의 titanic 데이터로 그래프 만들기
df = sns.load_dataset('titanic')
# print(df)

# sns.countplot(data=df, x = 'sex')
# plt.show()

### 다양한 파라미터(옵션) 사용해보기
# sns.countplot(data=df, x='class')
# sns.countplot(data=df, x='class', hue='alive')  # hue 옵션 : 변수의 항목별로 막대의 색을 다르게 표현하는 파라미터다.
# sns.countplot(data=df, y = 'class', hue='alive')
# plt.show()

''' 모듈(Module) 
-> 어떤 패키지는 함수가 너무 많이 때문에 비슷한 함수끼리 몇 개의 모듈로 나뉘어 있다
-> 패키지 안의 작은 꾸러미
'''

# import matplotlib.pyplot as plt
# plt.show()

### 모듈에 들어있는 함수를 사용하려면 "패키지명.모듈명.함수명()" 를 입력하면 된다
# import sklearn.metrics
# sklearn.metrics.accuracy_score()

### 모듈명.함수명() 으로 함수 사용하기
# import sklearn.metrics
# from sklearn import metrics
#
# metrics.accuracy_score()

### 함수명() 으로 함수 사용하기 - from 패키지명.모듈명 import 함수명
# from sklearn.metrics import accuracy_score
# accuracy_score()

### 패키지 설치 - pydataset 패키지 설치 예제
'''
파이참 왼쪽 하단 Terminal -> Command prompt ( 아나콘다 가상환경 접속 ) -> 설치 명령어

파이썬에서 설치를 담당하는 명령어는 pip 다.

pip install 패키지명
'''

### 아나콘다 가상환경 다루기 - 아나콘다 명령어

## 1. 현재 내 환경에 존재하는 패키지 조회 - Command Prompt - conda list

# ----------------------------------------------------------------------------------------------------------------------
'''
데이터 프레임 : 데이터를 다룰 때 가장 많이 사용하는 데이터 형태로, 행과 열로 구성된 사각형 모양의 표 처럼 생긴 자료형

1. 열 은 속성이다. 열 은 컬럼 또는 변수라 불린다
2. 행 은 한 사람의 정보이다. 행은 로우(row) 또는 케이스(case)라 불린다.
3. 한 사람의 정보는 가로 한 줄에 나열된다
4. 하나의 단위가 하나의 행이 된다
5. 데이터 크다 = 행 또는 열이 많다
'''

### 데이터 프레임 생성
''' pandas : 데이터 가공 패키지 '''
import pandas as pd

df = pd.DataFrame({
    'name' : ['김지훈', '이유진', '박동현', '김민지'],
    'english' : [90,80,60,70],
    'math' : [50,60,100,20]
})

# print(df)
'''
딕셔너리 : 파이썬에서 {키(key) : 값(value)}의 쌍으로 구성되는 자료 구조
        -> 중괄호 {} 로 표현한다
'''
'''
파이썬에서 코드 맨 앞에 공백이 들어가면 오류가 난다 -> IndentationError : unexpected indent 들여쓰기 오류
파이썬에서는 들여쓰기(공백 4칸)로 코드의 우선순위를 구분한다

파이참을 쓸 때 상위코드에서 오류가 나면 하위코드가 실행이 되지 않는다
-> 실행시킬 때마다 1번부터 끝 코드까지 한번에 실행되기 때문
'''

### 데이터 프레임으로 분석하기

# 1. 특정 변수의 값 추출하기
a = df['english']
# print(a)

# 2. 변수의 값으로 합계 구하기
a = sum(df['english'])
# print('영어점수 합계 =============>' , a)

# 3. 변수의 값으로 평균 구하기
mean = a/4
# print('영어 점수 평균 : ',mean)
# print('수학 점수 평균 : ', sum(df['math'])/4)

# ----------------------------------------------------------------------------------------------------------------------
''' 외부 데이터 활용 '''

### 엑셀 데이터 활용
path = 'C:/Python_Project_hdh/Data/' # 경로 맨 뒤에 / 붙여놓기!

# 1. 엑셀 파일 로드
df_exam = pd.read_excel(path + 'excel_exam.xlsx')
# print(df_exam)

# 2. 분석
a = sum(df_exam['english'])/len(df_exam)
# print(a)
s = sum(df_exam['science'])/20
# print('과학 점수 평균 : ' , s)

''' 파이썬에서 자료형의 길이를 구하는 함수 -> len() '''

x = [1,2,3,4,5,6]
# print(len(x))

### 엑셀의 첫 행이 변수명이 아닌 경우
df_exam_novar = pd.read_excel(path + 'excel_exam_novar.xlsx', header=None)
# print(df_exam_novar)

### 시트가 여러개인 경우
df_exam_2 = pd.read_excel(path + 'excel_exam.xlsx', sheet_name= 1)
''' sheet_name 에 숫자로 불러오는 경우 제일 처음 시트의 인덱스 번호는 0번 부터다. '''
# print(df_exam_2)

### csv파일 - 각 값들이 쉼표로 구분되어 있는 파일 (엑셀 연동)
df_csv_exam = pd.read_csv(path + 'exam.csv')
# print(df_csv_exam)

# ----------------------------------------------------------------------------------------------------------------------
''' 데이터 파악 , 다루기 쉽게 수정 '''
''' 
- 데이터 파악 함수
1. head() : 앞 부분 출력
2. tail() : 뒷 부분 출력
3. shape : 행, 열 개수 출력
4. info() : 변수 속성 출력
5. describe() : 요약 통계량 출력
'''

exam = pd.read_csv(path + 'exam.csv')
# print(exam)

# 1. 데이터의 앞 부분 확인
a = exam.head()   # 앞 부분 5행까지 출력
# print(a)

# 2. 데이터의 뒷 부분 확인
b = exam.tail()   # 뒷 부분 5행까지 출력
# print(b)

# 3. 데이터 행 열 개수 확인
# print(exam.shape)

# 4. 변수 속성 파악
# print(exam.info())
''' int64 : 정수형 '''

# 5. 요약 통계량
# print(exam.describe())

### mpg 데이터 파악
mpg = pd.read_csv(path + 'mpg.csv')
# print(mpg)
# print(mpg.info())
''' object : 문자열
    float64 : 실수형 '''
# print(mpg.describe())
# print(mpg.describe(include='all'))      # 다 보고 싶을 때

''' unique : 고유값 빈도 = 중복을 제거한 범주의 개수 
    top : 최빈값 = 개수가 가장 많은 값
    freq : 최빈값 빈도 = 개수가 가장 많은 값의 개수 '''

''' 함수와 메서드의 차이 '''
'''
        sum()       pd.read_csv()       df.head()
        내장 함수   패키지 함수          메서드
        
1. 내장 함수 : 가장 기본적인 함수 형태, 함수 이름과 괄호를 입력해서 사용한다. 파이썬에 내장되어 있는 함수
2. 패키지 함수 : 패키지 이름을 먼저 입력하고 접속 연산자(.)를 찍고 함수 이름과 괄호를 입력해서 사용한다
3. 메서드(Method) : 변수가 지니고 있는 함수, 변수명을 입력한 다음 점을 찍고 메서드의 이름과 괄호를 입력해서 사용한다.
4. 어트리뷰트 : 변수가 지니고 있는 값, 출력하려면 변수명 뒤에 점을 찍고 어트리뷰트 이름을 입력하면 된다
    ex) df.shape
'''

### 변수명 바꾸기 ***

df_raw = pd.DataFrame({
    'var1' : [1,2,1],
    'var2' : [2,3,2]
})
# print(df_raw)

''' 변수명을 바꾸기 전에 복사본을 생성한다.
    -> 원본 데이터의 손실을 방지하기 위해 '''

### 데이터 복사본 생성
df_new = df_raw.copy()
# print(df_new)

df_new = df_new.rename(columns={'var2' : 'v2'})     # var2를 v2로 수정
# print(df_new)

''' Q1. mpg 데이터를 불러와서 복사본을 만드세요. (변수명 mpg_new)
    Q2. 복사본 데이터를 사용해서 cty를 city로 변경하고 출력'''
mpg = pd.read_csv(path + 'mpg.csv')
mpg_new = mpg.copy()
mpg_new = mpg_new.rename(columns={'cty' : 'city'})
# print(mpg_new)

### 파생변수 생성 : 기존의 변수를 사용해서 변형해 만든 변수

''' cty 도시연비와 hwy 고속도로 연비의 통합 연비 변수 생성 '''

mpg = pd.read_csv(path + 'mpg.csv')

mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2
# print(mpg.head())

# 1. 통합 연비 변수 평균
total_mean = sum(mpg['total']) / len(mpg)
# print('통합 연비 평균 : ',total_mean)

# print(mpg['total'].mean())

# 2. 조건문을 활용해서 파생변수 생성 - 조건문 함수
''' 연비 기준을 충족해 고연비 합격 판정 - 몇 대나 될까? '''

# 2-1 기준값 정하기
# print(mpg['total'].describe())
# mpg['total'].plot.hist()
# plt.show()

# 2-2 합격 판정 변수 생성
import numpy as np      # numpy : 배열 연산, 통계치 계산 등 수치 연산을 할 떄 자주 사용하는 패키지

''' 20 이상이면 pass, 그렇지 않으면 fail 부여
    np.where(조건, 조건에 할당할 때, 조건에 해당하지 않을때) '''
mpg['test'] = np.where(mpg['total'] >= 20, 'pass', 'fail')
# print(mpg.head(20))

# 2-3 빈도표 합격 판정 자동차 수 확인
a = mpg['test'].value_counts()      # value_counts() : 빈도표 생성 함수
# print(a)

# 2-4 막대 그래프로 빈도 표현
# count_test = mpg['test'].value_counts()
# count_test.plot.bar(rot = 0)    # rot = 0 : 축 이름을 수평으로 만들기
# plt.show()

# 3. 중첩 조건문을 활용
''' 연비 등급 변수 만들기
    A 등급 : 30 이상
    B 등급 : 20 - 29
    C 등급 : 20 미만 '''

# 3-1 total 기준으로 A,B,C 부여
mpg['grade'] = np.where(mpg['total'] >= 30, 'A'
                        ,np.where(mpg['total'] >= 20, 'B', 'C'))
# print(mpg.head(20))

# 3-2 빈도표와 막대그래프로 표현
# count_grade = mpg['grade'].value_counts()
# count_grade.plot.bar(rot=0)
# plt.show()

# 3-3 알파벳 순으로 막대 정렬
count_grade = mpg['grade'].value_counts().sort_index()      # 메서드 체이닝 : 변수에 메서드를 순서대로 적용하는 기법
# print(count_grade)
# count_grade.plot.bar(rot=0)
# plt.show()

### 목록에 해당하는 행으로 변수 생성 - compact, suvcompact, 2seater 인 경우에는 small 부여
# print(mpg.head())
''' | (버티컬 바) : 또는 (or)을 의미하는 기호이다'''
''' 여러 조건을 입력할 때는 각 조건을 괄호로 감싸준다.'''
mpg['size'] = np.where((mpg['category'] == 'compact') |
                       (mpg['category'] == 'suvcompact') |
                       (mpg['category'] == '2seater'),
                       'small', 'large')
# print(mpg['size'].value_counts())

# ----------------------------------------------------------------------------------------------------------------------
''' 데이터 전처리 - 원하는 형태로 데이터 가공 작업 '''

'''
- 데이터 전처리 함수
1. query() : 행 추출
2. 데이터명[] : 열 추출
3. sort_values() : 정렬
4. groupby() : 집단별로 나누기
5. assign() : 변수 추가
6. agg() : 통계치 구하기
7. merge() : 데이터 합치기(열 기준)
8. concat() : 데이터 합치기(행 기준)
'''

### 조건에 맞는 데이터만 추출 - df.query()

import pandas as pd
exam = pd.read_csv(path + 'exam.csv')
# print(exam)

# 1. 1반만 추출하기
a = exam.query('nclass == 1')
# print(a)

# 2. 1반만 제외하고 추출
a = exam.query('nclass != 1')   # ! : 부정의 의미
# print(a)

# 3. 초과, 미만, 이상, 이하 조건

# 3-1. 수학 점수가 50점을 초과한 경우
a = exam.query('math > 50')
# print(a)

# 3-2. 영어 점수가 80점 이하
a = exam.query('english <= 80')
# print(a)

# 4. 여러 조건을 만족하는 행 추출 - 그리고(and) - &
# 4-1. 2반이면서 영어점수가 80점 이상인 경우
a = exam.query('nclass == 2 & english >= 80')
# print(a)

# 5. 여러 조건 중 하나 이상 충족하는 행 추출 - 또는(or) - |
# 5-1. 수학 점수가 90점 이상이거나, 영어 점수가 90점 이상인 경우
a = exam.query('math >= 90 | english >= 90')
# print(a)

# 6. 목록에 해당하는 행 추출
# 6-1. 1반, 3반, 4반인 경우만 추출
a = exam.query('nclass == 1 | nclass == 3 | nclass == 4')
a = exam.query('nclass in [1,3,4]')
print(a)