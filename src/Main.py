import pandas as pd
import numpy as np
import math

def Method_1(Data_inlying, Data_len_inlying):
    alpha = 1 # location sensitivity parameter
    gamma = 0.2 # similarity parameter
    R_hat = 0.5 # reputation
    chi = 0 # centroid index
    Dc = 0.05 # location diameter
    Psi = 1 # punishment factor
    B = 500 # Budget
    
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
    lambda_tm = 1
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
            chi = i
    print("Centroid:")
    print(chi)
    data_kai = Data_inlying[chi][index_tem]
    print(data_kai)

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
        Theta[i] = math.exp(-Dc*alpha)*(1-math.exp(-loc_dist[i][chi]*alpha)) # get location factor
        Tb[i] = min((1-Theta[i])*Lambda_r,1) # get base trust
        Tf[i] = Tb[i]*(1+delta[i]) # final trust
        level[i] = Tf[i] - R_hat # feedback level
#        sum_temp += Tf[i]
        if Tf[i] < 0.5:
            phi[i] = 0
        else:
            phi[i] = math.log(Tf[i]+1)

    sum_Tf = phi.sum()
    for i in range(Data_len_inlying):
        if Tf[i] < 0.5:
            phi[i] = 0
        else:
            phi[i] = phi[i]/sum_Tf * B
    np.savetxt("OutputMethod1_ad2.csv",phi,delimiter=',')
    #print("Base trust:")
    #print(Tb)
    #print("Similarity factor:")
    #print(delta)
    #print("Final trust:")
    #print(Tf[0:49])
    print("Rewards:")
    print(phi)
    
def Method_2(Data_inlying, Data_len_inlying):
    alpha = 1 # location sensitivity parameter
    gamma = 0.2 # similarity parameter
    R_hat = 0.5 # reputation
    kai = 0 # centroid index
    Dc = 0.05  # location diameter
    Psi = 1.2 # punishment factor
    B = 500 # Budget
    
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
    lambda_tm = 1
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

    np.savetxt("OutputMethod2_ad2.csv",phi,delimiter=',')
    #print("Distance:")
    #print(dist)
    #print("Distance factor:")
    #print(Theta)
    #print("Base trust:")
    #print(Tb)
    #print("Similarity factor:")
    #print(delta)
    #print("Final trust:")
    #print(Tf)
    print("Rewards:")
    print(phi)
    
def Method_3(Data_inlying, Data_len_inlying, Data_len):
    # Remuneration Method 3
    epsilon = 0.05
    sum_quality = 0
    B = 500
    index_ID = 0
    index_data = 1
    index_Time = 2
    index_lat = 3
    index_lon = 4
    index_tem = 5
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
    chi = 0 
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
            chi = i
    print("Centroid:")
    print(chi)
    data_chi = Data_inlying[chi][index_tem]
    print(data_chi)
    for i in range(Data_len_inlying):
        if i!=chi:
            theta[i] = dist[i][chi] / data_chi
            quality[i] = 1 / (theta[i] + epsilon)
            sum_quality += quality[i]
        

    avg_quality_frac = 1.0 / (Data_len_inlying-1)
    quality[chi] = 0.0
    quality[chi] = np.max(quality)
    sum_quality += quality[chi]
    print("Theta:")
    print(theta)
    #print(avg_quality_frac)
    #print(sum_quality)
    for i in range(Data_len_inlying):
        quality_frac[i] = quality[i] / sum_quality
        reward1[i] = B_m + B_n * (quality_frac[i] - avg_quality_frac)
   
    np.savetxt("OutputMethod3_ad2.csv",reward1,delimiter=',')
    print("Distance:")
    print(dist)
    #print("Fraction quality:")
    #print(quality_frac)
    #print(reward1)

WhichMethod = 3
index_ID = 0
index_data = 1
index_Time = 2
index_lat = 3
index_lon = 4
index_tem = 5
r_threshold = 5
frac_threshold = 0.75
B = 500

#Read data from csv file
df=pd.read_csv('crowd_temperature_origin_ad3.csv', sep=',',header=None,skiprows=1)
dataset = df.to_numpy()
#dataset = dataset[0:99,:]
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
        if i>99:
            print(k)
        k+=1

Data_len_inlying = k;
#print(Data_inlying[0:(Data_len_inlying-1),:])
print("Inlying dataset size:")
print(Data_len_inlying)
print(Data_inlying[71,:])
print(Data_inlying[72,:])
print(Data_inlying[73,:])
if WhichMethod == 1:
    Method_1(Data_inlying, Data_len_inlying)
elif WhichMethod ==2:
    Method_2(Data_inlying, Data_len_inlying)
elif WhichMethod ==3:
    # Remuneration Method 3
    Method_3(Data_inlying, Data_len_inlying, Data_len)
