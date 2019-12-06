# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path

#Code starts here 
data=pd.read_csv(path)
data.Gender.replace('-','Agender',inplace=True)
gender_count=data.Gender.value_counts()
gender_count.plot(kind='bar',figsize=(15,10))


# --------------
#Code starts here
alignment=data.Alignment.value_counts()
plt.figure(figsize=(6,6))
plt.pie(alignment)
plt.title('Character Alignment')



# --------------
#Code starts here
import seaborn as sb
sc_df=data[['Strength','Combat']]

sc_covariance=(sc_df.Strength.cov(sc_df.Combat))
print(sc_covariance)
#sc_strength=data.Strength.std()
sc_strength=sc_df.loc[:,"Strength"].std()
sc_combat=sc_df.loc[:,"Combat"].std()
#sc_combat=data.Combat.std()

sc_pearson = sc_covariance/(sc_combat*sc_strength)
print(sc_pearson)

ic_df=data[['Intelligence','Combat']]
ic_covariance=(data.Intelligence.cov(data.Combat))
ic_intelligence=data.Intelligence.std()
ic_combat=data.Combat.std()
ic_pearson=ic_covariance/(ic_combat*ic_intelligence)
#ic_pearson = ic_df.corr(method='pearson')
print(ic_pearson)




# --------------
#Code starts here
total_high=data['Total'].quantile(q=0.99)
print(total_high)
super_best=data[data['Total']>total_high]
#print(super_best)
super_best_names=list(super_best['Name'])
print(super_best_names)



# --------------
#Code starts here


ax_1=data.boxplot(column=['Intelligence'])
ax_2=data.boxplot(column=['Speed'])
ax_3=ax_1=data.boxplot(column=['Power'])


