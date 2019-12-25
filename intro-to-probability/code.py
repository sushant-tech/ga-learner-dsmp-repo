# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=df[df['fico'].astype(float)>700].shape[0]/df.shape[0]
p_b=df['purpose'].value_counts().debt_consolidation/df.shape[0]
#df1=df[df['purpose']=='dept_consolidation'].value_counts()
df1=df[df['purpose'] == 'debt_consolidation']

p_a_b=df1[df1['fico'].astype(float)>700].shape[0]/df1.shape[0]
result=p_a_b==p_a
print(result)
# code ends here


# --------------
# code starts here
#prob_lp=df[df['paid.back.loan']=='Yes'].value_counts()
prob_lp=df['paid.back.loan'].value_counts().Yes/df.shape[0]
prob_cs=df['credit.policy'].value_counts().Yes/df.shape[0]
new_df=df[df['paid.back.loan']=='Yes']
prob_pd_cs=new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]
bayes=prob_pd_cs*prob_lp/prob_cs
print(bayes)

# code ends here


# --------------
# code starts here
df1=df[df['paid.back.loan']=='No']
df.purpose.value_counts(normalize=True).plot(kind='bar')


# code ends here


# --------------
# code starts here
inst_median=df.installment.median()
inst_mean=df.installment.mean()
df['installment'].hist(normed=True,bins=50)
df['log.annual.inc'].hist(normed=True,bins=50)

# code ends here


