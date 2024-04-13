import pandas as pd
import seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

path = 'C:/bigdata/'
직종별 = pd.read_excel('직종별.xlsx')

# print(직종별)

matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

df = pd.DataFrame(직종별)

filtered_df = df[df['시점'].str.contains('부산')]

busan_data = filtered_df.drop(columns=['시점'])

busan_data = busan_data.apply(lambda x: x.replace(',', '') if isinstance(x, str) else x).astype(int)


column_means = busan_data.mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=column_means.index, y=column_means.values)
plt.title('부산 직종별 평균')
plt.xlabel('직종')
plt.ylim(0,70000)
plt.tight_layout()
plt.show()