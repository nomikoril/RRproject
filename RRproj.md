---
title: "Credit Scoring Classification Analysis"
author: "Joachim Ndhokero- 4408804 and Nomin Batbayar - 442627"
date: today
format: 
  html:
    toc: true
    toc-depth: 3
    toc-title: Contents
    toc-location: left
    theme: minty
    fontsize: 1.1em
    linestretch: 1.7
execute:
  echo: fenced
title-block-banner: true 
code-fold: true
keep-md: true
---


## Introduction

Credit scoring is very crucial part in the financial industry. It can be very helpful for lenders to assess how the borrowers are worth in terms of credit. And it is equally important that steps and processes taken to build credit scoring model should be open and transparent. This way one can ensure that the model is reliable and fair.
In this project, we develop a credit scoring midel using 3 machine learning approaches, to assess which model better predicts the Credit scores. 

Firstly, we describe the dataset used in our study. We obtained Credit scoring dataset from [kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification). Which includes customer's demographics, credit history, income and other relavant features. To ensure transparency we provided detailed data preprocessinf steps that we performed, and steps for resolving missing and outlier values.

Then, to find most relavant features to our model, we use correlation coefficients to decide if we include variable in model for numeric variables. For categorical variable we used ANOVA to asses if variable has association with credit scoring dependent variable.

Next we use .... machine learning algorithms. We specify the hyperparameters, model training processes, to enable other researchesr to replicate our model development process.

##### Importing Neccessary Libraries

Following libraries are used for this project

::: {.cell execution_count=1}
```` { .cell-code}
```{{python}}
import pandas as pd
import re
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
```

````
:::


##### Dataset

Initial dimension of the data had 100'000 rows and 28 variables. 

-   "ID" - Unique identifier
-   "Customer_ID" - Unique identifier of customer
-   "Name" - Name of person
-   "SSN" - Unique identifier
-   "Month" - Month of the year
-   "Age" - Age of the person, We limited it into 14+56
-   "Occupation" - Occupation of the person, 16 factor
-   "Annual_Income" - Annual income of the person, continuous variable
-   "Monthly_Inhand_Salary" - Monthly salary, continuous variable
-   "Num_Bank_Accounts" - Num of bank accounts the person holds
-   "Num_Credit_Card" - Num of credit card the person holds
-   "Interest_Rate" - Interest rate on credit card
-   "Num_of_Loan" - Num of loans from the bank
-   "Type_of_Loan" - Type of loan
-   "Delay_from_due_date" - Average number of days delayed from the payment date
-   "Num_of_Delayed_Payment" - Average number of payments delayed by a person
-   "Changed_Credit_Limit" - Percentage change in credit card limit
-   "Num_Credit_Inquiries" - number of credit card inquiries
-   "Credit_Mix" - Classification of the mix of credits
-   "Outstanding_Debt" - remaining debt to be paid (in USD)
-   "Credit_Utilization_Ratio" - utilization ratio of credit card
-   "Credit_History_Age" - age of credit history of the person
-   "Payment_of_Min_Amount" - whether only the minimum amount was paid by the person
-   "Total_EMI_per_month" - monthly EMI payments (in USD)
-   "Amount_invested_monthly" - monthly amount invested by the customer (in USD)
-   "Payment_Behaviour" - payment behavior of the customer
-   "Monthly_Balance" - monthly balance amount of the customer (in USD)
-   "Credit_Score" - bracket of credit score (Poor, Standard, Good)

Our target variable Credit_Score has 3 classes, but we will only work with **"Poor"** and **"Good"** categories to further analysis. So that we are left with **46'826** observations.

::: {.cell execution_count=2}
```` { .cell-code}
```{{python}}
#| warning: false
os.getcwd()
credit_score=pd.read_csv('train.csv')
credit_score=credit_score.drop(['ID','Name','SSN'], axis=1) # remove unnecessary unique identifier columns
credit_score=credit_score[credit_score.Credit_Score!='Standard'] # filter target variable and drop Standard category
print(credit_score.shape)
credit_score.head()
```

````

::: {.cell-output .cell-output-stdout}
```
(46826, 25)
```
:::

::: {.cell-output .cell-output-display execution_count=2}

```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer_ID</th>
      <th>Month</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>Annual_Income</th>
      <th>Monthly_Inhand_Salary</th>
      <th>Num_Bank_Accounts</th>
      <th>Num_Credit_Card</th>
      <th>Interest_Rate</th>
      <th>Num_of_Loan</th>
      <th>...</th>
      <th>Credit_Mix</th>
      <th>Outstanding_Debt</th>
      <th>Credit_Utilization_Ratio</th>
      <th>Credit_History_Age</th>
      <th>Payment_of_Min_Amount</th>
      <th>Total_EMI_per_month</th>
      <th>Amount_invested_monthly</th>
      <th>Payment_Behaviour</th>
      <th>Monthly_Balance</th>
      <th>Credit_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUS_0xd40</td>
      <td>January</td>
      <td>23</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>_</td>
      <td>809.98</td>
      <td>26.822620</td>
      <td>22 Years and 1 Months</td>
      <td>No</td>
      <td>49.574949</td>
      <td>80.41529543900253</td>
      <td>High_spent_Small_value_payments</td>
      <td>312.49408867943663</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUS_0xd40</td>
      <td>February</td>
      <td>23</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>NaN</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>Good</td>
      <td>809.98</td>
      <td>31.944960</td>
      <td>NaN</td>
      <td>No</td>
      <td>49.574949</td>
      <td>118.28022162236736</td>
      <td>Low_spent_Large_value_payments</td>
      <td>284.62916249607184</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUS_0xd40</td>
      <td>March</td>
      <td>-500</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>NaN</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>Good</td>
      <td>809.98</td>
      <td>28.609352</td>
      <td>22 Years and 3 Months</td>
      <td>No</td>
      <td>49.574949</td>
      <td>81.699521264648</td>
      <td>Low_spent_Medium_value_payments</td>
      <td>331.2098628537912</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUS_0xd40</td>
      <td>April</td>
      <td>23</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>NaN</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>Good</td>
      <td>809.98</td>
      <td>31.377862</td>
      <td>22 Years and 4 Months</td>
      <td>No</td>
      <td>49.574949</td>
      <td>199.4580743910713</td>
      <td>Low_spent_Small_value_payments</td>
      <td>223.45130972736786</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUS_0xd40</td>
      <td>May</td>
      <td>23</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>Good</td>
      <td>809.98</td>
      <td>24.797347</td>
      <td>22 Years and 5 Months</td>
      <td>No</td>
      <td>49.574949</td>
      <td>41.420153086217326</td>
      <td>High_spent_Medium_value_payments</td>
      <td>341.48923103222177</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>
```

:::
:::


## Data Preprocessing

We prepared the data set for further analysis step by transforming character variables into categorical/factor variables, and numeric variables, treating missing values, and standardized it and treating outliers.

We have noticed there is some strange values like "_",  "!@9#%8",  "#F%$D@*&8" in dataset. You can check the coding part below for  details.

::: {.cell execution_count=3}
```` { .cell-code}
```{{python}}
def replace_weird(credit_score): # create function for remove _ syntax
    if credit_score is np.NaN or not isinstance(credit_score, str):
        return credit_score
    else:
        return str(credit_score).strip('_ ,"')
credit_score = credit_score.applymap(replace_weird).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.NaN) # replace weird strings with na
```

````
:::


### Treating Missing Data 

Below we can see that there is quite many missing data in our dataset. Removing them would result significant drop in our number of datasets. So that we need to try treating it as much as we can.

::: {.cell execution_count=4}
```` { .cell-code}
```{{python}}
credit_score.isna().sum()
```

````

::: {.cell-output .cell-output-display execution_count=4}
```
Customer_ID                    0
Month                          0
Age                            0
Occupation                  3274
Annual_Income                  0
Monthly_Inhand_Salary       7115
Num_Bank_Accounts              0
Num_Credit_Card                0
Interest_Rate                  0
Num_of_Loan                    0
Type_of_Loan                4780
Delay_from_due_date            0
Num_of_Delayed_Payment      3257
Changed_Credit_Limit        1000
Num_Credit_Inquiries         902
Credit_Mix                  9491
Outstanding_Debt               0
Credit_Utilization_Ratio       0
Credit_History_Age          4206
Payment_of_Min_Amount          0
Total_EMI_per_month            0
Amount_invested_monthly     2155
Payment_Behaviour           3608
Monthly_Balance              595
Credit_Score                   0
dtype: int64
```
:::
:::


Below shows that we have dataset which includes credit scoring history of 8'692 unique customers January to August.  Using Customer_Id, we can replace some feature's missing value of certain customers, if other values are not missing.

::: {.cell execution_count=5}
```` { .cell-code}
```{{python}}
num_customer= len(set(credit_score["Customer_ID"] ))
print("Number of unique customer: ", num_customer)
num_month= set(credit_score["Month"] )
print("Number of month: ", num_month)
```

````

::: {.cell-output .cell-output-stdout}
```
Number of unique customer:  8692
Number of month:  {'July', 'August', 'March', 'May', 'June', 'February', 'January', 'April'}
```
:::
:::


For example, for Customer_ID=CUS_0x2dbc 3 of the value is missing. So instead of removing them we replaced the missing values with other most occured non missing value.

::: {.cell execution_count=6}
```` { .cell-code}
```{{python}}
credit_score[credit_score.Customer_ID=='CUS_0x2dbc'].groupby('Customer_ID')['Occupation'].apply(list)
```

````

::: {.cell-output .cell-output-display execution_count=6}
```
Customer_ID
CUS_0x2dbc    [nan, Engineer, nan, Engineer, nan, Engineer]
Name: Occupation, dtype: object
```
:::
:::


After replacing the missiong value of occupation, above example looks like this.

::: {.cell execution_count=7}
```` { .cell-code}
```{{python}}
credit_score['Occupation'] = credit_score['Occupation'].fillna(credit_score.groupby('Customer_ID')['Occupation'].transform(lambda x: x.fillna(stats.mode(x)[0][0])))
credit_score[credit_score.Customer_ID=='CUS_0x2dbc'].groupby('Customer_ID')['Occupation'].apply(list)
```

````

::: {.cell-output .cell-output-display execution_count=7}
```
Customer_ID
CUS_0x2dbc    [Engineer, Engineer, Engineer, Engineer, Engin...
Name: Occupation, dtype: object
```
:::
:::


With this method we also replaced missing values of categorical variables, Credit_Mix, Payment_Behaviour, Type_of_Loan.

::: {.cell execution_count=8}
```` { .cell-code}
```{{python}}
credit_score['Credit_Mix'] = credit_score['Credit_Mix'].fillna(credit_score.groupby('Customer_ID')['Credit_Mix'].transform(lambda x: x.fillna(stats.mode(x)[0][0])))
credit_score['Payment_Behaviour'] = credit_score['Payment_Behaviour'].fillna(credit_score.groupby('Customer_ID')['Payment_Behaviour'].transform(lambda x: x.fillna(stats.mode(x)[0][0])))
credit_score['Type_of_Loan'] = credit_score['Type_of_Loan'].fillna(credit_score.groupby('Customer_ID')['Type_of_Loan'].transform(lambda x: x.fillna(stats.mode(x)[0][0])))
# for credit history age we replaced with pervious or next month's value
credit_score['Credit_History_Age'] = credit_score.groupby('Customer_ID')['Credit_History_Age'].apply(lambda x: x.interpolate().bfill().ffill())
```

````
:::


As for numerical variables, we take average of other non missing values.

::: {.cell execution_count=9}
```` { .cell-code}
```{{python}}
credit_score[credit_score.Customer_ID=='CUS_0x1018'].groupby('Customer_ID')['Num_of_Delayed_Payment'].apply(list)
```

````

::: {.cell-output .cell-output-display execution_count=9}
```
Customer_ID
CUS_0x1018    [22, 22, 22, 20, nan, 22, 22, 22]
Name: Num_of_Delayed_Payment, dtype: object
```
:::
:::


After replacing the missiong value of Num_of_Delayed_Payment, above example looks like this.

::: {.cell execution_count=10}
```` { .cell-code}
```{{python}}
credit_score['Num_of_Delayed_Payment'] = credit_score['Num_of_Delayed_Payment'].fillna(credit_score.groupby('Customer_ID')['Num_of_Delayed_Payment'].transform(lambda x: x.fillna(x.astype('float64').mean())))

credit_score[credit_score.Customer_ID=='CUS_0x1018'].groupby('Customer_ID')['Num_of_Delayed_Payment'].apply(list)
```

````

::: {.cell-output .cell-output-display execution_count=10}
```
Customer_ID
CUS_0x1018    [22, 22, 22, 20, 21.714285714285715, 22, 22, 22]
Name: Num_of_Delayed_Payment, dtype: object
```
:::
:::


And with this method, we do same for Monthly_Inhand_Salary, Changed_Credit_Limit, Num_Credit_Inquiries, Amount_invested_monthly, Monthly_Balance variables.

::: {.cell execution_count=11}
```` { .cell-code}
```{{python}}
credit_score['Monthly_Inhand_Salary'] = credit_score['Monthly_Inhand_Salary'].fillna(credit_score.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(lambda x: x.fillna(x.astype('float64').mean())))
credit_score['Changed_Credit_Limit'] = credit_score['Changed_Credit_Limit'].fillna(credit_score.groupby('Customer_ID')['Changed_Credit_Limit'].transform(lambda x: x.fillna(x.astype('float64').mean())))
credit_score['Num_Credit_Inquiries'] = credit_score['Num_Credit_Inquiries'].fillna(credit_score.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(lambda x: x.fillna(x.mean())))
credit_score['Amount_invested_monthly'] = credit_score['Amount_invested_monthly'].fillna(credit_score.groupby('Customer_ID')['Amount_invested_monthly'].transform(lambda x: x.fillna(x.astype('float64').mean())))
credit_score['Monthly_Balance'] = credit_score['Monthly_Balance'].fillna(credit_score.groupby('Customer_ID')['Monthly_Balance'].transform(lambda x: x.fillna(x.astype('float64').mean())))
```

````
:::


After missing value treatment we have now left with very few missing value.

::: {.cell execution_count=12}
```` { .cell-code}
```{{python}}
credit_score.isna().sum()
```

````

::: {.cell-output .cell-output-display execution_count=12}
```
Customer_ID                  0
Month                        0
Age                          0
Occupation                   0
Annual_Income                0
Monthly_Inhand_Salary       85
Num_Bank_Accounts            0
Num_Credit_Card              0
Interest_Rate                0
Num_of_Loan                  0
Type_of_Loan                 0
Delay_from_due_date          0
Num_of_Delayed_Payment      33
Changed_Credit_Limit         6
Num_Credit_Inquiries         5
Credit_Mix                   0
Outstanding_Debt             0
Credit_Utilization_Ratio     0
Credit_History_Age          47
Payment_of_Min_Amount        0
Total_EMI_per_month          0
Amount_invested_monthly     19
Payment_Behaviour            0
Monthly_Balance              9
Credit_Score                 0
dtype: int64
```
:::
:::


### Data type
Here we are converting variables into suitable data types numeric or categorical.

::: {.cell execution_count=13}
```` { .cell-code}
```{{python}}
credit_score["Month"] = credit_score["Month"].astype("category")
credit_score['Age']=credit_score['Age'].astype('int')
credit_score.Occupation=credit_score.Occupation.astype('category')
credit_score.Annual_Income=credit_score.Annual_Income.astype('float')
credit_score.Monthly_Inhand_Salary=credit_score.Monthly_Inhand_Salary.astype('float')
credit_score.Num_Bank_Accounts=credit_score.Num_Bank_Accounts.astype('float')
credit_score.Num_Credit_Card=credit_score.Num_Credit_Card.astype('float')
credit_score.Interest_Rate=credit_score.Interest_Rate.astype('float')
credit_score.Num_of_Loan=credit_score.Num_of_Loan.astype('int')
credit_score.Changed_Credit_Limit=pd.to_numeric(credit_score.Changed_Credit_Limit, errors='coerce')
credit_score.Num_Credit_Inquiries=pd.to_numeric(credit_score.Num_Credit_Inquiries, errors='coerce')
credit_score.Credit_Mix=credit_score.Credit_Mix.astype('category')
credit_score.Outstanding_Debt=pd.to_numeric(credit_score.Outstanding_Debt, errors='coerce')
credit_score.Payment_of_Min_Amount=credit_score.Payment_of_Min_Amount.astype('category')
credit_score.Amount_invested_monthly=pd.to_numeric(credit_score.Amount_invested_monthly, errors='coerce')
credit_score.Credit_Score=credit_score.Credit_Score.astype('category')
credit_score.Monthly_Balance=pd.to_numeric(credit_score.Monthly_Balance, errors='coerce')
```

````
:::


