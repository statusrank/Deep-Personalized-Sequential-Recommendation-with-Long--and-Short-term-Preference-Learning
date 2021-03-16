import seaborn as sns
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
'''
        HR@10,MRR@10,NDCG@10,HR@20,MRR@20,NDCG@20
        GRU:
            0.4581,0.2680,0.3089,0.5268,0.2733,0.3256
        PGRU:
            0.5688,0.3440,0.3921,0.6432,0.3496,0.4092
		SPLU: 
            0.4546,0.2751,0.3144,0.5280,0.2812,0.3321
		DPSR:
            0.5839,0.3522,0.4023,0.6605,0.3567,0.4184
data = {
    'metric': ["HR@10","MRR@10","NDCG@10","HR@20","MRR@20","NDCG@20"]
}
'''

name_list = ['Monday','Tuesday','Friday','Sunday','WWWW','JJJJ']
num_list = [1.5,0.6,7.8,6,5,6]
num_list1 = [1,2,3,1,2,3]
num_list2 = [1,1,1,1,4,5]
x =list(range(len(num_list)))
total_width, n = 1.2, 3
width = total_width / n
plt.bar(x, num_list, width=width, label='boy',fc = 'y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='girl',tick_label = name_list,fc = 'r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='hh',tick_label = name_list,fc = 'b')
plt.legend()
plt.show()