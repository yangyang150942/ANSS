import pandas as pd
import numpy as np
import math

def Method_1(Data_inlying, Data_len_inlying):
    alpha = 1 # location sensitivity parameter
    gamma = 0.3 # similarity parameter
    R_hat = 1 # reputation
    kai = 0 # centroid index
    Dc = 0.05  # location diameter
    Psi = 1 # punishment factor
    B = 10000 # Budget
    
    index_ID = 0
    index_data = 1
    index_Time = 2
    index_lat = 3
    index_lon = 4
    index_tem = 5
    sum_temp = 0
    sum_temp_2 = 0
    sum_min = float("inf")
    dist = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object) #Temperature distance between each pair of data
    dist_square = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    loc_dist = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object) #Location distance
    loc_dist_square = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    Theta = np.empty(shape=(Data_len_inlying,1),dtype=object)
    delta = np.empty(shape=(Data_len_inlying,1),dtype=object)
    S = np.empty(shape=(Data_len_inlying,Data_len_inlying),dtype=object)
    sum_S = np.empty(shape=(Data_len_inlying,1),dtype=object)
    lambda_dt = 1
    lambda_tm = 0.94
    Lambda_r = lambda_dt * lambda_tm
    Tb = np.empty(shape=(Data_len_inlying,1),dtype=object) #base trust score
    Tf = np.empty(shape=(Data_len_inlying,1),dtype=object) #final trust score
    level = np.empty(shape=(Data_len_inlying,1),dtype=object) #feedback level
    phi = np.empty(shape=(Data_len_inlying,1),dtype=object) #rewards
    
    #Calculate distance between each pair of data based on temperature
    for i in range(Data_len_inlying):
        for j in range(Data_len_inlying):
            dist_square[i][j] = (Data_inlying[i][index_tem]-Data_inlying[j][index_tem])*(Data_inlying[i][index_tem]-Data_inlying[j][index_tem])
            dist[i][j] = math.sqrt(distance_square[i][j])
            loc_dist_square[i][j] = (Data_inlying[i][index_lat]-Data_inlying[j][index_lat])*(Data_inlying[i][index_lat]-Data_inlying[j][index_lat]) + (Data_inlying[i][index_lon]-Data_inlying[j][index_lon])*(Data_inlying[i][index_lon]-Data_inlying[j][index_lon])
            loc_dist[i][j] = math.sqrt(loc_dist_square[i][j])
            
    #Find the centroid data
    for i in range(Data_len_inlying):
        sum_temp = 0
        for j in range(Data_len_inlying):
            sum_temp += dist_square[i][j]

        if sum_temp < sum_min:
            sum_min = sum_temp
            kai = i
    data_kai = Data_inlying[kai][index_tem]
    print(data_kai)

#    print(min_dist)
#    print(max_dist)

    #Prepare for getting similarity factor
    min_dist = np.min(dist)
    max_dist = np.max(dist)
    for i in range(Data_len_inlying):
        sum_temp = 0
        for j in range(Data_len_inlying):
            S[i][j] = -(-1 + (1-(-1))/(max_dist-min_dist)*(dist[i][j]-min_dist))
            if j!=i:
                sum_temp += S[i][j]
        
        sum_S[i] = sum_temp
    
    sum_temp = 0  
    # Calculate the final trust
    for i in range(Data_len_inlying):
        delta[i] = (sum_S[i] / (Data_len_inlying - 1)) * math.exp(-1/Data_len_inlying)*gamma # get similarity factor
        Theta[i] = math.exp(-Dc*alpha)*(1-math.exp(-loc_dist[i][kai]*alpha)) # get location factor
        Tb[i] = min(R_hat*(1-Theta[i])*Lambda_r,1) # get base trust
        Tf[i] = Tb[i]*(1+delta[i]) # final trust
        level[i] = Tf[i] - R_hat # feedback level
#        sum_temp += Tf[i]
        if Tf[i] < 0.5:
            phi[i] = 0
        else:
            phi[i] = math.log(Tf[i]+1)
#           if i!=0:
#                phi[i] = math.log(sum_temp) - math.log(sum_temp_2)
#            else:
#                phi[i] = math.log(Tf[i])
#        sum_temp_2 += Tf[i]
    sum_Tf = phi.sum()
    for i in range(Data_len_inlying):
        if Tf[i] < 0.5:
            phi[i] = 0
        else:
            phi[i] = phi[i]/sum_Tf
    print("Base trust:")
    print(Tb)
    print("Similarity factor:")
    print(delta)
    print("Final trust:")
    print(Tf)
    print("Rewards:")
    print(phi)
    
def Method_2(Data_inlying, Data_len_inlying):
    alpha = 1 # location sensitivity parameter
    gamma = 0.3 # similarity parameter
    R_hat = 1 # reputation
    kai = 0 # centroid index
    Dc = 0.05  # location diameter
    Psi = 1 # punishment factor
    B = 10000 # Budget
    
    index_ID = 0
    index_data = 1
    index_Time = 2
    index_lat = 3
    index_lon = 4
    index_tem = 5
    sum_temp = 0
    sum_min = float("inf")
    dist = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object) #Temperature distance between each pair of data
    dist_square = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    loc_dist = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object) #Location distance
    loc_dist_square = np.empty(shape=(Data_len_inlying,Data_len_inlying), dtype=object)
    Theta = np.empty(shape=(Data_len_inlying,1),dtype=object)
    delta = np.empty(shape=(Data_len_inlying,1),dtype=object)
    S = np.empty(shape=(Data_len_inlying,Data_len_inlying),dtype=object)
    sum_S = np.empty(shape=(Data_len_inlying,1),dtype=object)
    lambda_dt = 1
    lambda_tm = 0.94
    Lambda_r = lambda_dt * lambda_tm
    Tb = np.empty(shape=(Data_len_inlying,1),dtype=object) #base trust score
    Tf = np.empty(shape=(Data_len_inlying,1),dtype=object) #final trust score
    level = np.empty(shape=(Data_len_inlying,1),dtype=object) #feedback level
    phi = np.empty(shape=(Data_len_inlying,1),dtype=object) #rewards
    
    #Calculate distance between each pair of data based on temperature
    for i in range(Data_len_inlying):
        for j in range(Data_len_inlying):
            dist_square[i][j] = (Data_inlying[i][index_tem]-Data_inlying[j][index_tem])*(Data_inlying[i][index_tem]-Data_inlying[j][index_tem])
            dist[i][j] = math.sqrt(distance_square[i][j])
            loc_dist_square[i][j] = (Data_inlying[i][index_lat]-Data_inlying[j][index_lat])*(Data_inlying[i][index_lat]-Data_inlying[j][index_lat]) + (Data_inlying[i][index_lon]-Data_inlying[j][index_lon])*(Data_inlying[i][index_lon]-Data_inlying[j][index_lon])
            loc_dist[i][j] = math.sqrt(loc_dist_square[i][j])
            
    #Find the centroid data
    for i in range(Data_len_inlying):
        sum_temp = 0
        for j in range(Data_len_inlying):
            sum_temp += dist_square[i][j]

        if sum_temp < sum_min:
            sum_min = sum_temp
            kai = i
    data_kai = Data_inlying[kai][index_tem]
    print(data_kai)

#    print(min_dist)
#    print(max_dist)

    #Prepare for getting similarity factor
    min_dist = np.min(dist)
    max_dist = np.max(dist)
    for i in range(Data_len_inlying):
        sum_temp = 0
        for j in range(Data_len_inlying):
            S[i][j] = -(-1 + (1-(-1))/(max_dist-min_dist)*(dist[i][j]-min_dist))
            if j!=i:
                sum_temp += S[i][j]
        
        sum_S[i] = sum_temp
                
    # Calculate the final trust
    for i in range(Data_len_inlying):
        delta[i] = (sum_S[i] / (Data_len_inlying - 1)) * math.exp(-1/Data_len_inlying)*gamma # get similarity factor
        Theta[i] = math.exp(-Dc*alpha)*(1-math.exp(-loc_dist[i][kai]*alpha)) # get location factor
        Tb[i] = min(R_hat*(1-Theta[i])*Lambda_r,1) # get base trust
        Tf[i] = Tb[i]*(1+delta[i]) # final trust
        level[i] = Tf[i] - R_hat # feedback level
    
    # Calculate the rewards for each report
    sum_Tf = Tf.sum()
    for i in range(Data_len_inlying):
        if level[i] < 0:
            phi[i] = Tf[i]/sum_Tf * B * math.exp(level[i]*Psi)
        else:
            phi[i] = Tf[i]/sum_Tf * B
    print("Distance:")
    print(dist)
    print("Distance factor:")
    print(Theta)
    print("Base trust:")
    print(Tb)
    print("Similarity factor:")
    print(delta)
    print("Final trust:")
    print(Tf)
    print("Rewards:")
    print(phi)
    
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

WhichMethod = 1
index_ID = 0
index_data = 1
index_Time = 2
index_lat = 3
index_lon = 4
index_tem = 5
r_threshold = 5
frac_threshold = 0.75
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

if WhichMethod == 1:
    Method_1(Data_inlying, Data_len_inlying)
elif WhichMethod ==2:
    Method_2(Data_inlying, Data_len_inlying)
elif WhichMethod ==3:
    # Remuneration Method 3
    Method_3(Data_inlying, Data_len_inlying)



