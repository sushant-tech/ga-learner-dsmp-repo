# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'

#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
np.array(new_record)
#Code starts here

data=np.genfromtxt(path,delimiter=",",skip_header=1)
print(data)
census=np.array(data)
census=np.concatenate((new_record,data),axis=0)
print(census)

#data=np.concatenate((new_record),axis=0)
#census=np.array(data)



# --------------
#Code starts here
age=np.array(census[:,0])
print(age)
max_age=age.max()
print(max_age)
min_age=age.min()
print(min_age)
age_mean=np.mean(age)
print(age_mean)
age_std=np.std(age)
print(age_std)



# --------------
#Code starts here
race_0=census[census[:,2]==0]
race_1=census[census[:,2]==1]
race_2=census[census[:,2]==2]
race_3=census[census[:,2]==3]
race_4=census[census[:,2]==4]

len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)

MM=min(len_0,len_1,len_2,len_3,len_4)
print(MM)

minority_race=3
print(minority_race)


# --------------
#Code starts here
senior_citizens=census[census[:,0]>60]
#working_hours=census[:,6]
working_hours_sum=senior_citizens.sum(axis=0)[6]
senior_citizens_len=len(senior_citizens)
avg_working_hours=(working_hours_sum)/(senior_citizens_len)
print(avg_working_hours)



# --------------
#Code starts here
high=census[census[:,1]>10]
low=census[census[:,1]<=10]
avg_pay_high=high.mean(axis=0)[7]
print(avg_pay_high)
avg_pay_low=low.mean(axis=0)[7]
print(avg_pay_low)

#np.high_equals(avg_pay_high,avg_pay_low)
#avg_pay_low=mean()


