import pandas as pd
import numpy as np
import math

def Method_2(Data_inlying, Data_len_inlying):
    alpha = 1
    beta = 1
    gamma = 1
    R_hat = 1 #reputation
    

def Method_3(Data_inlying, Data_len_inlying):
    # Remuneration Method 3
    epsilon = 0.05
    sum_quality = 0
    dist = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    dist_square = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    theta = np.empty(shape=(Data_len_inlying,1),dtype=object)
    quality = np.empty(shape=(Data_len_inlying,1),dtype=object)
    quality_frac = np.empty(shape=(Data_len_inlying,1),dtype=object)
    reward1 = np.empty(shape=(Data_len_inlying,1),dtype=object)
    B_m = B / Data_len_inlying
    B_n = B / Data_len
    sum_temp = 0
    sum_min = float("inf")
    kai = 0 
    for i in range(Data_len_inlying):
        for j in range(Data_len_inlying):
            dist_square[i][j] = (Data_inlying[i][index_tem]-Data_inlying[j][index_tem])*(Data_inlying[i][index_tem]-Data_inlying[j][index_tem])
            dist[i][j] = math.sqrt(distance_square[i][j])
        
    for i in range(Data_len_inlying):
        sum_temp = 0
        for j in range(Data_len_inlying):
            sum_temp += dist_square[i][j]
    
        if sum_temp < sum_min:
            sum_min = sum_temp
            kai = i

    print(kai)
    data_kai = Data_inlying[kai][index_tem]
    print(data_kai)
    for i in range(Data_len_inlying):
        theta[i] = dist[i][kai] / data_kai
        quality[i] = 1 / (theta[i] + epsilon)
        sum_quality += quality[i]

    avg_quality_frac = 1.0 / Data_len_inlying  
    print(theta)
    print(avg_quality_frac)
    print(sum_quality)
    for i in range(Data_len_inlying):
        quality_frac[i] = quality[i] / sum_quality
        reward1[i] = B_m + B_n * (quality_frac[i] - avg_quality_frac)
        
    print(quality_frac)
    print(reward1)
    print(np.amax(reward1))
    print(np.amin(reward1))  

WhichMethod = 3
index_ID = 0
index_data = 1
index_Time = 2
index_lat = 3
index_lon = 4
index_tem = 5
r_threshold = 5
frac_threshold = 0.8
B = 10000

#Read data from csv file
df=pd.read_csv('..\crowd_temperature.csv', sep=',',header=None,skiprows=1)
dataset = df.to_numpy()
Data_len = dataset[:,0].size
Data_len_inlying = 0

#Outlier detection
num_threshold = Data_len*frac_threshold
i = 0
j = 0
k = 0
cnt = 0
distance = np.empty(shape=(Data_len,Data_len), dtype=object)
distance_square = np.empty(shape=(Data_len,Data_len), dtype=object)
Data_inlying = np.empty(shape=(Data_len,dataset[0,:].size), dtype=object)

for i in range(Data_len):
    cnt=0;
    for j in range(Data_len):
        distance_square[i][j] = (dataset[i][index_tem]-dataset[j][index_tem])*(dataset[i][index_tem]-dataset[j][index_tem])
        distance[i][j] = math.sqrt(distance_square[i][j])
        if distance[i][j] < r_threshold:
            cnt+=1
    
    if cnt > num_threshold:
        Data_inlying[k] = dataset[i]
        k+=1

Data_len_inlying = k;
print(Data_inlying[0:(Data_len_inlying-1),:])
print(Data_len_inlying)

#if WhichMethod == 1:
    
#elif WhichMethod ==2:

if WhichMethod ==3:
    # Remuneration Method 3
    Method_3(Data_inlying, Data_len_inlying)



