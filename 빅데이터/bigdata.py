import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

path = 'C:/bigdata/'
exam = pd.read_excel('exam.xlsx')

# print(exam)

exam['전입자-전출자(서울)'] = (exam['총전입(서울)']) - (exam['총전출(서울)'])
print(exam['전입자-전출자(서울)'])

exam['전입자-전출자(부산)'] = (exam['총전입(부산)']) - (exam['총전출(부산)'])
print(exam['전입자-전출자(부산)'])