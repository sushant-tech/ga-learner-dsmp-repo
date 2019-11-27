# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 


# code starts here
bank=pd.read_csv(path)
categorical_var=bank.select_dtypes(include='object')
print(categorical_var)

numerical_var=bank.select_dtypes(include='number')
print(numerical_var)




# code ends here


# --------------
# code starts here
bank.drop(['Loan_ID'],inplace=True,axis=1)
banks=pd.DataFrame(bank)
banks.isnull().sum()
# #print(banks)
bank_mode=banks.mode()
# print(bank_mode)
banks.fillna('bank_mode',inplace=True)
print(banks)

# print(banks)
#code ends here
#nba["College"].fillna("No College", inplace = True) 
  


# --------------
# Code starts here
import numpy as np
import pandas as pd



avg_loan_amount = banks.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)


print (avg_loan_amount)




# avg_loan_amount=pd.pivot_table(banks,values='LoanAmount',index=["Gender","Married","Self_Employed"],
#                         aggfunc=np.mean)
# print(avg_loan_amount)


# code ends here



# --------------
# code starts here





loan_approved_se=banks[(banks.Self_Employed=='Yes')&(banks.Loan_Status=='Y')]['Loan_Status'].count()

loan_approved_nse=banks[(banks.Self_Employed=='No')&(banks.Loan_Status=='Y')]['Loan_Status'].count()


Loan_Status=banks.Loan_Status.count()

percentage_se=(loan_approved_se/Loan_Status)*100

percentage_nse=(loan_approved_nse/Loan_Status)*100
print(percentage_se)
print(percentage_nse)

#code ends here



# --------------
# code starts here
loan_term=banks['Loan_Amount_Term'].apply(lambda x: int(x)/12)
print(loan_term)

big_loan_term=len(loan_term[loan_term>=25])
print(big_loan_term)

# code ends here
# loan_approved_se=banks[(banks.Self_Employed=='Yes')&(banks.Loan_Status=='Y')]['Loan_Status'].count()



# --------------
# code starts here





loan_groupby=banks.groupby('Loan_Status')['ApplicantIncome','Credit_History']
print(loan_groupby)
mean_values=loan_groupby.agg(np.mean)




# code ends here


