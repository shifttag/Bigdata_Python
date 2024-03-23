''' 24 - 03 - 23 강의 자료 '''
import time

import requests

web = requests.get('https://www.daangn.com/fleamarket/')
# print(web.text)

''' 
BeautifulSoup 라이브러리 : HTML 문서에서 원하는 부분만 추출하게 해주는 기능 
'''
from bs4 import BeautifulSoup

soup = BeautifulSoup(web.content, 'html.parser')
# print(soup.h1)  # . 을 찍어서 접속을 하면 태그 접속을 의미한다.

## 1. ul 태그의 하위 항목을 모두 뽑아오고 싶을 떄
# for child in soup.ul.children:
#     print(child)

''' 파이썬에서 하위 코드는 들여쓰기 (띄어쓰기 4칸) 으로 구분을 한다. '''

## 2. find_all() : 지정 태그의 모든 값을 가져오는 함수, 결과값을
# print(soup.find_all('h2'))
#
# for i in soup.find_all('h2'):
#     print('현재 요소 >> ',i)

## 2-1. 정규식 활용 방법 - <ol> 이든 <ul> 이든 다 포함된 리스트를 긁어오고 싶을 때
# import re
# for f in soup.find_all(re.compile("[ou]l")):
#     print(f)

## 2-2. 리스트 활용 방법 - 원하는 태그를 직접 지정해서 뽑는 경우, ex) h1, p 만 보고싶다
# for f in soup.find_all(['h1','p']):
#     print(f)

## 2-3. HTML 속성 활용 - 속성을 지정해서 뽑고 싶을 때
# a = soup.find_all(attrs={'class':'card-title'})
# for i in a:
#     print('매물명 :',i.text)

## 2-4. CSS 선택자를 통해 원하는 부분 가지고 오기 - select()
# a = soup.select(".card-region-name")
# print(a)

# a = soup.select("#hot-articles-head-title") # id 이용할 때
# print(a)

## 2-5. 텍스트만 읽어오고 싶을 때
# for x in range(0,10):
#     print(soup.select('.card-title')[x].get_text())

# ----------------------------------------------------------------------------------------------------------------------
''' 네이버 날씨 요약 프로그래밍 - 웹 크롤링 '''

### import
import datetime
from bs4 import BeautifulSoup
import urllib.request
import requests

## 현재 시간을 출력하고 본인 스타일에 맞게 출력문 수정
now = datetime.datetime.now()   # 현재 시간
# print(now)
nowDate = now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초 입니다.')
# print(nowDate)
print('■'*100)
print('\t\t\t\t\t\t\t\t ※ Python Web Crawling Project ※')
print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t 관리자 : hdhadmin')
print('■'*100)
print('반갑습니다,','현재 시간은',nowDate, '\n')  # \n : 줄 바꿈
print("\t Let Me Summarise Today's Info! \n")

### 서울 날씨
print('#오늘의 #날씨 #요약 \n')
webpage = urllib.request.urlopen('https://search.naver.com/search.naver?where=nexearch&sm=top_sug.asiw&fbm=0&acr=1&acq=%EC%84%9C%EC%9A%B8&qdt=0&ie=utf8&query=%EC%84%9C%EC%9A%B8+%EB%82%A0%EC%94%A8')
soup = BeautifulSoup(webpage, 'html.parser')
temps = soup.find('strong','')  # 온도
# print(temps.text)
cast = soup.find('p','summary') # 날씨
# print(cast.text)
print('--> 서울 날씨 : ', temps.get_text(), cast.get_text())

webpage_b = urllib.request.urlopen('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EB%B6%80%EC%82%B0+%EB%82%A0%EC%94%A8&oquery=%EC%84%9C%EC%9A%B8+%EB%82%A0%EC%94%A8&tqi=iQHgcdqVN8VssLb1K%2FNssssssuR-185659')
soup_b = BeautifulSoup(webpage_b, 'html.parser')
temps_b = soup_b.find('strong', '') # 부산 온도
cast_b = soup_b.find('p', 'summary')# 부산 날씨
print('--> 부산 날씨 : ', temps_b.get_text(), cast_b.get_text())

webpage_j = urllib.request.urlopen('https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all&query=%EC%A0%9C%EC%A3%BC+%EB%82%A0%EC%94%A8&oquery=%EB%B6%80%EC%82%B0+%EB%82%A0%EC%94%A8&tqi=iQHhvsqpts0sssjPGsGssssssUh-147648')
soup_j = BeautifulSoup(webpage_j, 'html.parser')
temps_j = soup_j.find('strong', '') # 제주 온도
cast_j = soup_j.find('p', 'summary')# 제주 날씨
print('--> 제주도 날씨 : ', temps_j.get_text(), cast_j.get_text())














































































