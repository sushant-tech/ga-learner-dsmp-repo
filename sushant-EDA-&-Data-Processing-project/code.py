# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data=pd.read_csv(path)
data['Rating'].hist(bins=10)
data=data[data['Rating']<=5]
data.hist(bins=10)
#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
#print(total_null)
percent_null=(total_null/data.isnull().count())
missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])
print(missing_data)
data=data.dropna(subset=['Current Ver','Android Ver'])
total_null_1=data.isnull().sum()
percent_null_1=(total_null_1/data.isnull().count())
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
a=sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10)
plt.xticks(rotation = 90)
a.set_titles('Rating vs Category[Boxplot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].astype(int)

le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
data['Installs'].unique()
ax=sns.regplot(x="Installs",y="Rating",data=data)
plt.title('Rating vs Installs [Regplot]')
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype(float)
sns.regplot(x="Price",y="Rating",data=data)
plt.title('Rating vs Price [Regplot]')
#Code ends here


# --------------

#Code starts here
print(data['Genres'].unique())
data['Genres']=data['Genres'].str.split(';').str[0]
gr_mean=data[['Genres','Rating']].groupby(['Genres'],as_index=False).mean()
print(gr_mean.describe())
gr_mean=gr_mean.sort_values('Rating')

print(gr_mean.head(1))

print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here
#print(data['Last Updated'])
data['Last Updated']=pd.to_datetime(data['Last Updated'])

max_date=data['Last Updated'].max()
print(max_date)
diff=max_date-pd.to_datetime(data['Last Updated'])
data['Last Updated Days']=diff.dt.days


sns.regplot(x="Last Updated Days",y="Rating",data=data)
plt.title("Rating vs Last Updated [RegPlot]")
#Code ends here


