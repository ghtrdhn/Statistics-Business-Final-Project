# Optional
# for filter warning
import warnings
warnings.simplefilter("ignore")

# load data
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
sns.set_theme(style="whitegrid")

# modelling
import statsmodels.formula.api as smf
from scipy.special import expit, logit

def print_coef_std_err(results):
    """
    Function to combine estimated coefficients and standard error in one DataFrame
    :param results: <statsmodels RegressionResultsWrapper> OLS regression results from 
    :return df: <pandas DataFrame> 
    """
    coef = results.params
    std_err = results.bse
    
    df = pd.DataFrame(data = np.transpose([coef, std_err]), 
                      index = coef.index, 
                      columns=["coef","std err"])
    return df

# import bank_personal_loan dataset
bank_loan = pd.read_csv('bank_loan.csv')
df = pd.DataFrame(bank_loan)
df

# drop ID and ZIP Code, and other unneeded dataset for this evaluation columns
bank_loan.drop(['ID', 'ZIP Code','Securities Account', 'CD Account', 'Online','CreditCard'], axis=1, inplace=True)
bank_loan

bank_loan = bank_loan.rename(columns = {"Personal Loan": "PersonalLoan"})

#In the dataset, it is possible that there is a null value,duplicate data, and inconsistent or invalid data format, for a better decision making and improve data evaluation output, the dataset has to be cleaned from null values and inconsistency format. The following data cleaning procedure are conducted in Jupyter Lab, and the process are as follows:

bank_loan

#Data Cleaning
#The dataset consists of 5000 rows and 8 columns, and certain inconsistent format are already could be seen from this dataset which is the CCAvg variable, further information is required to clean the dataset.

bank_loan.info()

#Cek data missing value
nan_col = bank_loan.isna().sum().sort_values(ascending = False)
# ada 875 data missing value dalam 3 kolom

# Mendapatkan persentase missing value tiap kolom
n_data = len(bank_loan)

percent_nan_col = (nan_col/n_data) * 100
percent_nan_col

# Create Data Info dataset, tipe data, apakah tedapat missing values
list_item = []
for col in bank_loan.columns:
    list_item.append([col, bank_loan[col].dtype, bank_loan[col].isna().sum(), 100*bank_loan[col].isna().sum()/len(bank_loan[col]), bank_loan[col].nunique(), bank_loan[col].unique()[:4]])
desc_df = pd.DataFrame(data=list_item, columns='feature data_type null_num null_pct unique_num unique_sample'.split())
desc_df

#It seems that the / sign in column CCAvg represents a decimal. Therefore, we first correct the CCAvg column by replace . instead of / and then convert type of CCAvg to float64:

bank_loan['CCAvg'] = df['CCAvg'].str.replace('/', '.').astype('float64')
bank_loan

# Create Data Info dataset, tipe data, apakah tedapat missing values
list_item = []
for col in bank_loan.columns:
    list_item.append([col, bank_loan[col].dtype, bank_loan[col].isna().sum(), 100*bank_loan[col].isna().sum()/len(bank_loan[col]), bank_loan[col].nunique(), bank_loan[col].unique()[:4]])
desc_df = pd.DataFrame(data=list_item, columns='feature data_type null_num null_pct unique_num unique_sample'.split())
desc_df

#It seems that we have a negative value in the Experience column, which is illogical, so since we do not have access to the owner of the data, we assume that the negative data was actually positive, so we convert the negative numbers into positive ones.

# find negative values in Experience columns
bank_loan[bank_loan['Experience'] < 0]

# convert above 52 rows to positive value
bank_loan[bank_loan['Experience'] < 0] = bank_loan[bank_loan['Experience'] < 0].abs()
bank_loan

### Overview of Data
The overview for the descriptive statstics is given as follows.

pd.set_option('display.max_rows', 20) # for show all rows
round(bank_loan.describe(include = 'all').T, 2)

#- The average income the customer is about 73.77 thousands dollar.
#- The average of credit card spending of the customer is about 1.94 thousands dollar.
#- The average of customer age is about 45 years old.
#- The average family member of customer is about 2 $\approx$ 3.
#- The average education is 1 $\approx$ 2, that means bachelor and master degree.

bank_loan.groupby(["Education"])[["Age","Income","CCAvg","Experience"]].mean()
#According to customer education score:
#- on average , the age of customer with undergraduate degree is 44 $\approx$ 45 years old, and customer with professional degree is 46 years old.
#- on average, the highest income is from undergraduate degree with 85 $\approx$ 86 thousand dollar, and the lowest income is from graduate degree with 64 $\approx$ 65 thousand dollar.
#- on average, follows the income score, the highest credit card spending also from undergraduate studies with 2 $\approx$ 3 thousand dollar, and the lowest credit card spending also from graduate degree with 1 $\approx$ 2 thousand dollar.
#- on average, the customer most experience level is from professional degree with 20 $\approx$ 21 years, and the least experience level is from graduate degree from 19 $\approx$ 20 years.

bank_loan.groupby(["Family"])[["Age","Income","CCAvg","Experience"]].mean()
#According to customer family member:

#- on average, the youngest customer age is from 2 family member with 45 $\approx$ 46 years old, and the oldest is from 3 family member with 46 $\approx$ 47 years old.
#- on average, the higest income is from 2 family member with 84 $\approx$ 85 thousand dollar, and the lowest income is from 4 family member with 62 $\approx$ 63 thousand dollar.
#- on average, following the income level, the the highest credit card spending is from 2 family member with 2 $\approx$ 3 thousand dollar, and the lowest credit card spending is from 4 family member with 1 $\approx$ 2 thousand dollar.
#- on average the most experienced level is from 3 family member with 20 $\approx$ 21 years, and the least experienced level is from 4 family member with 18 $\approx$ 19 years.

bank_loan.info()

custom_palette = sns.color_palette("icefire",2)
sns.pairplot(data = bank_loan, hue = "PersonalLoan", palette = custom_palette)
#- The plot shows the distribution of personal loan in each variable, the line structure of income show the person who approve personal loan is higher income compare to person whos does not approve personal loan, also apply to experience variable, which people that with higher experience tend to approve personal loan.
#- The customer who have higher credit card average spending, and higher mortgage value are higher to approve personal loan, compare to customer who decline personal loan.

col = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'PersonalLoan']
#Heatmap correlation metrics

plt.figure(figsize=(12, 10))

heatmap = sns.heatmap(bank_loan[col].corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("coolwarm", as_cmap=True))

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
#The heatmap correlation, indicate:

#- The age and experience variable are perfectly positive correlation with value is 0.99
#- The income and average credit card spending are moderate positive correlation with value 0.65
#- The Personal Loan and income have a moderate positive correlation with value 0.5
#- Next, followed by Personal Loan variable with average credit card spending has a weak positive relationship with 0.37.

# check distribution Scatter matrix (splom) with go.Splom
sns.set_palette('icefire',2)
fig, ax = plt.subplots(3,3,figsize=(12,20))
#fig.suptitle('Histogram of Bank Loan', fontsize = 20)
for i, col in enumerate(bank_loan):
    sns.histplot(bank_loan[col], kde=True, ax=ax[i//3, i%3])
plt.show()
#- The histogram plot identify that the variable are mostly skewed distribution with long right tail skewness such income, mortgage value, and credit card average spending (CCAvg).
#- Then for variable Age and experience, it tend to have a normal distribution.

#Visualize the data
bank_loancopy = bank_loan.copy()
bank_loancopy["PersonalLoan"] = ["Yes" if i==1 else "No" for i in bank_loan["PersonalLoan"]]

###### Bivariate Logistic Regression
### Data Preparation for Age

#Use LabelEncoder to convert the Default variable into numeric
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Age"]].head()

# Buat boxplot dengan seaborn 
custom_palette = sns.color_palette("icefire",2)
sns.boxplot(x = 'Age', y = 'PersonalLoan', data = bank_loancopy, palette = custom_palette)

#- The figure shows the distribution of Customer Age split by the binary Personal Loan variable.
#- Individuals who accept Personal Loan have higher level of minimum age and lower level of maximum age.

x = bank_loancopy["PersonalLoan"]
y = bank_loancopy["Age"]

# Plot the data
plt.scatter(x, y, color = "crimson", marker=".")

# Add a legend and labels

plt.xlabel("Personal Loan")
plt.ylabel("Age")

# Add a title and adjust the margins
plt.title("Personal Loan with Age")

# Show the plot
plt.show()

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_ageloan = smf.logit('PersonalLoan ~ Age', data = bank_loan).fit()
print(logit_model_ageloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Age", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)

results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-2.13 - 0.0022\text{Age})$$

#- Jadi dalam kasus diatas adalah , jika age (beta1) bertambah sebesar 1 year, maka terdapat pengurangan probability yang negatif sebesar 0.0022 perbedaan antara approve personal loan dan tidak approve.
#- Negative coefficient indicates that higher age tend to have lower probability of accepting personal loan.
#- To be precise, a one-unit change in age is associated with a negative difference in the log odds of accepting personal loan by 0.0022 unit.
#- Using divide by 4 rule: near the average age, 1 year unit more in age corresponds to an approximately 0.00055 % (0.00022/4) negative difference in probability of personal loan.

predictor = "Age"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Age")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- the sigmoidal shape of the fitted logistic regression curve, git can be seen that it is the same as the regression results,  that the variables in the bivariate logistic regression model for the personal loan variable with age, the fitted curve line variable, gets a slightly downward line, similar to the coefficient of intercept and age (beta 1) that has a negative value.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference age of 30 year (30 year & 60 year)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_age = np.array([30,60])
new_data = pd.DataFrame(data = new_age, columns = ["Age"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `30 year old` to accept personal loan is 9,90 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with age of `60 year` is lower, and equals 0.093 or 9,31 %.

### Data Preparation for Experience

#Use LabelEncoder to convert the Default variable into numeric

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Experience"]].head()

# Buat boxplot dengan seaborn
custom_palette = sns.color_palette("icefire",2)
sns.boxplot(x = 'Experience' , y = 'PersonalLoan', data = bank_loancopy, palette = custom_palette)

#- The figure shows the distribution of customer work experience split by the binary Personal Loan variable.
#- Individuals who accept Personal Loan have lower level of maximum experience in range.

y = bank_loancopy["Experience"]
x = bank_loancopy["PersonalLoan"]

# Plot the data
plt.scatter(x, y, color = "crimson", marker=".")

# Add a legend and labels

plt.xlabel("Personal Loan")
plt.ylabel("Experience")

# Add a title and adjust the margins
plt.title("Personal Loan with Experience")

# Show the plot
plt.show()

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_experienceloan = smf.logit('PersonalLoan ~ Experience', data = bank_loan).fit()
print(logit_model_experienceloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Experience", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-2.19 - 0.0024\text{Experience})$$

#- Negative coefficient indicates that higher experience tend to have lower probability of accepting personal loan.
#- To be precise, a one-unit change in experience is associated with a negative difference in the log odds of accepting personal loan by 0.0024 unit.
#- Using divide by 4 rule: near the average experience, 1 year unit more in experience corresponds to an approximately 0.0006 % (0.00024/4) negative difference in probability of personal loan.

predictor = "Experience"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Experience")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- the sigmoidal shape of the fitted logistic regression curve, it can be seen that it is the same as the regression results,  that the variables in the bivariate logistic regression model for the personal loan variable with experience, the fitted curve line variable, gets a slight downward line, similar to the coefficient of intercept and experience (beta 1) that has a negative value but has a lower value than the age variable.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference experience of 35 year (5 year & 40 year)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_experience = np.array([5,40])
new_data = pd.DataFrame(data = new_experience, columns = ["Experience"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `5 year` experience to accept personal loan is 9,92 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with experience of `40 year` is lower, and equals 0.091 or 9,17 %.

### Data Preparation for Income

#Use LabelEncoder to convert the Default variable into numeric

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Income"]].head()

# Buat boxplot dengan seaborn untuk total price
custom_palette = sns.color_palette("icefire",2)
sns.boxplot(x = 'Income' , y = 'PersonalLoan', data = bank_loancopy, palette = custom_palette)

#- The figure shows the distribution of customer income split by the binary Personal Loan variable.
#- Individuals who accept Personal Loan have higher level of minimum and maximum income in range.

y = bank_loancopy["Income"]
x = bank_loancopy["PersonalLoan"]

# Plot the data
plt.scatter(x, y, color = "crimson", marker=".")

# Add a legend and labels

plt.xlabel("Personal Loan")
plt.ylabel("Income")

# Add a title and adjust the margins
plt.title("Personal Loan with Income")

# Show the plot
plt.show()

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_incomeloan = smf.logit('PersonalLoan ~ Income', data = bank_loan).fit()
print(logit_model_incomeloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Income", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-6.12 + 0.0371\text{Income})$$

#- Jadi dalam kasus diatas adalah , jika Income (beta1) bertambah sebesar 1 (thousand dollar), maka terdapat pertambahan probability  yang positif sebesar 0.0371 perbedaan antara approve personal loan dan tidak approve.
#- Positive coefficient indicates that higher income tend to have higher probability of accepting personal loan.
#- To be precise, a one-unit change in income is associated with a positive difference in the log odds of accepting personal loan by 0.0371 unit.
#- Using divide by 4 rule: near the average income, 1 (thousand dollar) unit more in income corresponds to an approximately 0.9275 % (0.0371/4) positive difference in probability of personal loan.

predictor = "Income"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Income")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- The sigmoidal shape of the fitted logistic regression curve, the picture informed that the same as the results of the regression, that the variables in the bivariate logistic regression model for the personal loan variable with income, on the fitted curve line, get an upward line, similar to the coefficient of intercept and income (beta 1) that has a positive value and clarify taht variable income is one major predictor to personal loan.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference Income of 150 thousand dollar (50 thousand dollar & 200 thousand dollar)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_income = np.array([50,200])
new_data = pd.DataFrame(data = new_income, columns = ["Income"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `50 thousand dollar` income to accept personal loan is 1,3 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with income of `200 thousand dollar` is much higher, and equals 0.785 or 78,5 %.

### Data Preparation for CCAvg

#Use LabelEncoder to convert the Default variable into numeric

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Income"]].head()

# Buat boxplot dengan seaborn untuk total price
custom_palette = sns.color_palette("icefire",2)
sns.boxplot(x = 'CCAvg' , y = 'PersonalLoan', data = bank_loancopy, palette = custom_palette)

#- The figure shows the distribution of average credit carf spending split by the binary Personal Loan variable.
#- Individuals who accept Personal Loan have higher level of  maximum credit card average spending in range.

y = bank_loancopy["CCAvg"]
x = bank_loancopy["PersonalLoan"]

# Plot the data
plt.scatter(x, y, color = "crimson", marker=".")

# Add a legend and labels

plt.xlabel("Personal Loan")
plt.ylabel("CCAvg")

# Add a title and adjust the margins
plt.title("Personal Loan with CCAvg")

# Show the plot
plt.show()

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_ccavgloan = smf.logit('PersonalLoan ~ CCAvg', data = bank_loan).fit()
print(logit_model_ccavgloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ CCAvg", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-3.58 + 0.5115\text{CCAvg})$$

#- Positive coefficient indicates that higher credit card average spending tend to have higher probability of accepting personal loan.
#- To be precise, a one-unit change in credit card average spending is associated with a positive difference in the log odds of accepting personal loan by 0.5115 units.
#- Using divide by 4 rule: near the average credit card spending, 1 (thousand dollar) unit more in credit card average spending corresponds to an approximately 12.78% (0.5115/4) positive difference in probability of personal loan.

predictor = "CCAvg"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("CCAvg")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- The sigmoidal shape of the fitted logistic regression curve, the picture informed that the same as the results of the regression, that the variables in the bivariate logistic regression model for the personal loan variable with credit card average spending, on the fitted curve line, get an upward line, similar to the coefficient of intercept and income (beta 1) that has a positive value and it clarify that variable credit card average spending, as one product of bank is one major predictor to variable of approving personal loan.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference Credit Card Spending of 8 thousand dollar (2 thousand dollar & 10 thousand dollar)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_ccavg = np.array([2,10])
new_data = pd.DataFrame(data = new_ccavg, columns = ["CCAvg"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data
#- The model predict probability of a person that have `2 thousand dollar` credit card average spending to accept personal loan is 7,16 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with credit card average spending of `10 thousand dollar` is much higher, and equals 0.822 or 82,2 %.

### Data Preparation for Mortgage

#Use LabelEncoder to convert the Default variable into numeric
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Mortgage"]].head()

# Buat boxplot dengan seaborn
custom_palette = sns.color_palette("icefire",2)
sns.boxplot(x = 'Mortgage' , y = 'PersonalLoan', data = bank_loancopy, palette = custom_palette)
#- The figure shows the distribution of mortgage value split by the binary Personal Loan variable.
#- Individuals who accept Personal Loan have higher level of  maximum mortgage value in range.

y = bank_loancopy["Mortgage"]
x = bank_loancopy["PersonalLoan"]

# Plot the data
plt.scatter(x, y, color = "crimson", marker=".")

# Add a legend and labels

plt.xlabel("Personal Loan")
plt.ylabel("Mortgage")

# Add a title and adjust the margins
plt.title("Personal Loan with Mortgage")

# Show the plot
plt.show()

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_mortgageloan = smf.logit('PersonalLoan ~ Mortgage', data = bank_loan).fit()
print(logit_model_mortgageloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Mortgage", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

$$P(\text{Personal Loan}) = \text{logit}^{-1}(-2.50 + 0.0036\text{Mortgage})$$

#- Positive coefficient indicates that higher mortgage value tend to have higher probability of accepting personal loan.
#- To be precise, a one-unit change in education is associated with a positive difference in the log odds of accepting personal loan by 0.0036 units.
#- Using divide by 4 rule: near the average mortgage, 1 (thousand dollar) unit more in mortgage value corresponds to an approximately 0.09% (0.0036/4) positive difference in probability of personal loan.

predictor = "Mortgage"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Mortgage")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()
#- The sigmoidal shape of the fitted logistic regression curve, the picture informed that the same as the results of the regression, that the variables in the bivariate logistic regression model for the personal loan variable with mortgage value, on the fitted curve line, get an upward line, similar to the coefficient of intercept and income (beta 1) that has a positive value and it clarify that mortgage value as one product of bank, is one major predictor to variable of approving personal loan.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference Mortgage value of 500 thousand dollar (100 thousand dollar & 600 thousand dollar)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_mortgage = np.array([100,600])
new_data = pd.DataFrame(data = new_mortgage, columns = ["Mortgage"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `100 thousand dollar` mortgage value to accept personal loan is 10,4 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with mortgage value of `600 thousand dollar` is much higher, and equals 0.414 or 41,4 %.

### Data Preparation for Education

#Use LabelEncoder to convert the Default variable into numeric

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder Object and transform the Default variable
bank_loan["PersonalLoan"] = LabelEncoder().fit_transform(bank_loan["PersonalLoan"])

# Display the 5th first row after transforming
bank_loan[["PersonalLoan","Education"]].head()

sns.catplot(
    data=bank_loancopy, y="Education", palette = 'icefire', hue = 'PersonalLoan', kind = 'count')

#- It can be informed that customers tend to approve personal loans if they have a higher level of education, in this case it can be informed that customers with professional education are the highest in approving personal loans, and undergraduate is the lowest in approving personal loans.

# Kode di bawah bertujuan untuk melihat rata-rata dari botak_prob with kategori pekerjaan dengan menggunakan point plot.
fig, ax = plt.subplots(figsize = (8,5))
sns.pointplot(y = "Education", x = "PersonalLoan", data = bank_loancopy, ax = ax)
ax.set_title("Personal Loan with Education")
plt.show()

#- From the pointplot graph, it can be informed that customers with 1: undergraduate and 2: graduate education tend not to approve personal loans. Then, for customers with education 2: Graduate and 3: Professional tend to approve personal loans.

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_eduloan = smf.logit('PersonalLoan ~ Education', data = bank_loan).fit()
print(logit_model_eduloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Education", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-3.37 + 0.5548\text{Education})$$

#- Jadi dalam kasus diatas adalah , jika education(beta1) bertambah sebesar 1, maka terdapat pertambahan probability  yang positif sebesar 0.5548 perbedaan antara approve personal loan dan tidak approve.
#- Positive coefficient indicates that higher education tend to have higher probability of accepting personal loan.
#- To be precise, a one-unit change in education is associated with a positive difference in the log odds of accepting personal loan by 0.5548 units.
#- Using divide by 4 rule: near the average education, 1 unit more in Education corresponds to an approximately 13.87% (0.5548/4) positive difference in probability of personal loan.

predictor = "Education"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Education")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- The sigmoidal shape of the fitted logistic regression curve, the picture informed that the same as the results of the regression, that the variables in the bivariate logistic regression model for the personal loan variable with Education, on the fitted curve line, get an upward line, similar to the coefficient of intercept and income (beta 1) that has a positive value.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference education of 2 (Undergraduate & Professional)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_education = np.array([1,3])
new_data = pd.DataFrame(data = new_education, columns = ["Education"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `1 : Undergraduate` Education to accept personal loan is below 5,6 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with education of `3: Professional` is much higher, and equals 0.153 or 15.3 %.

### Data Preparation for Family

#Use LabelEncoder to convert the Personal Loan variable into numeric

sns.catplot(
    data=bank_loancopy, y="Family", palette = 'icefire', hue = 'PersonalLoan', kind = 'count')

#- It can be informed that customers tend to approve personal loans if they have a higher number of family member, in this case it can be informed that customers with 3 & 4 family member are the highest number in approving personal loans, and customer with 1 & 2 family member is the lowest in approving personal loans.

# Kode di bawah bertujuan untuk melihat rata-rata dari botak_prob with kategori pekerjaan dengan menggunakan point plot.
fig, ax = plt.subplots(figsize = (8,5))
sns.pointplot(y = "Family", x = "PersonalLoan", data = bank_loancopy, ax = ax)
ax.set_title("Personal Loan with Education")
plt.show()

#- From the pointplot graph, it can be informed that customers with 2 family member tend not to approve personal loans. Then, for customers with family member 3 family member tend to approve personal loans.

# Logistic Regression model with One predictor (numerical: Tenure)
logit_model_eduloan = smf.logit('PersonalLoan ~ Family', data = bank_loan).fit()
print(logit_model_eduloan.summary())

#Fungsi untuk menjalankan logistics regression model

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Family", bank_loan)

# Fit the model
model_default = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_default_coef = print_coef_std_err(model_default)
results_default_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-2.69 + 0.1807 \text{family})$$

#- Positive coefficient indicates that higher family member tend to have higher probability of accepting personal loan.
#- To be precise, a one-unit change in education is associated with a positive difference in the log odds of accepting personal loan by 0.1807 units.
#- Using divide by 4 rule: near the average family member, 1 unit more in family member corresponds to an approximately 4.51% (0.1807/4) positive difference in probability of personal loan.

predictor = "Family"
outcome = "PersonalLoan"
data = bank_loan
results_ = results_default_coef.copy()

# Plot the data
plt.scatter(data[predictor], data[outcome], marker= 'o', facecolors = "none", edgecolor="crimson", alpha=0.5, label='data')

# Calculate the fitted values
a_hat = results_.loc["Intercept"]["coef"]
b_hat = results_.loc[predictor]["coef"]

# get values from predictor range
x_range = np.linspace(np.min(data[predictor]), np.max(data[predictor]), 100)

# predicted probabilities of x in x_range
pred_prob = expit(a_hat + b_hat*x_range)

# Plot the fitted line
plt.plot(x_range, pred_prob, label="Fitted curve", color = "steelblue")

# Add a legend and labels
plt.legend()
plt.ylabel("Personal Loan")
plt.xlabel("Family")

# Add a title and adjust the margins
plt.title("Data and fitted regression line")

# Show the plot
plt.show()

#- The sigmoidal shape of the fitted logistic regression curve, the picture informed that the same as the results of the regression, that the variables in the bivariate logistic regression model for the personal loan variable with Family, on the fitted curve line, get an upward line, similar to the coefficient of intercept and income (beta 1) that has a positive value.

### Prediction

#- Next, make predict of personal loan for two individuals with a difference family of 3 (1 & 4 family member)

#dalam fungsi code kali ini, untuk mengetahui probaiblity apakah seseorang akan default atau tidak

new_family = np.array([1,4])
new_data = pd.DataFrame(data = new_family, columns = ["Family"])

new_data["predicted_p_personal_loan"] = model_default.predict(new_data)
new_data

#- The model predict probability of a person that have `1 Family member` Education to accept personal loan is below 7,5 %. 
#- In contrast, the predicted probability of approve personal loan for an individual with `4 Family member` is much higher, and equals 0.122 or 12.2 %.

### Multivariate Logistic Regression

# Logistic Regression with Multiple Predictors

logit_mode_multivariate = smf.logit('PersonalLoan ~ Age + Experience + Income + CCAvg + Mortgage + Education + Family', data = bank_loan).fit()
print(logit_mode_multivariate.summary())

# Create Logit model object
logit_model = smf.logit('PersonalLoan ~ Age + Experience + Income + CCAvg + Mortgage + Education + Family', bank_loan)

# Fit the model
model_personalLoan = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_personalLoan_coef = print_coef_std_err(model_personalLoan)
results_personalLoan_coef

#$$P(\text{Personal Loan}) = \text{logit}^{-1}(-12.80 - 0.0375 \text{Age} + 0.0499 \text{Experience} + 0.0540 \text{Income} + 0.1306 \text{CCAvg} + 0.0006 \text{Mortgage} + 1.6777 \text{Education} + 0.6869 \text{Family})$$

#- Positive coefficient indicates tend to have higher probability of accepting personal loan.
#- Negative coefficient indicates tend to have lower probability of accepting personal loan of each increase of the related variable
#- To be precise, a one-unit change in a positive coefficient is associated with a positive difference in the log odds of accepting personal loan.

#Comparing these coefficients, it would at first seem that Education and Family is a more important factor than any other predictors in determining the probability of approving personal loan. Such a statement is could be misleading, however we can evaluate it through the value of standard deviation:

print(f"standard deviation Age : {bank_loan['Age'].std():.2f}")
print(f"standard deviation Experience level : {bank_loan['Experience'].std():.2f}")
print(f"standard deviation Income : {bank_loan['Income'].std():.2f}")
print(f"standard deviation CCAvg : {bank_loan['CCAvg'].std():.2f}")
print(f"standard deviation Mortgage : {bank_loan['Mortgage'].std():.2f}")
print(f"standard deviation Education : {bank_loan['Education'].std():.2f}")
print(f"standard deviation Family : {bank_loan['Family'].std():.2f}")

#- The standard deviation of `Age` to the accepting personal loan is 11.46. The logistic regression coefficients corresponding to 1-standard-deviation differences are -0.0375 ∗ 11.46 = -0.4297 (Beta1* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Age corresponds to an expected -0.4297/4 or approximately `10.74% negative difference` in probability of approve personal loan.

#- The standard deviation of `Experience` to the accepting personal loan is 11.42. The logistic regression coefficients corresponding to 1-standard-deviation differences are 0.0499 ∗ 11.42 = 0.5698 (Beta2* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Experience corresponds to an expected 0.5698/4 or approximately `14.24% positive difference` in probability of approve personal loan.

#- The standard deviation of `Income` to the accepting personal loan is 46.03. The logistic regression coefficients corresponding to 1-standard-deviation differences are 0.0540 ∗ 46.03 = 2,4856 (Beta3* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Income corresponds to an expected 2.4856/4 or approximately `62.14% positive difference` in probability of approve personal loan. Remind that, the 1 unit in `Income` is equal to `1 thousand dollar`.

#- The standard deviation of `CCAvg` to the accepting personal loan is 1.75. The logistic regression coefficients corresponding to 1-standard-deviation differences are 0.1306 ∗  1.75 = 0.2285 (Beta4* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in CCAvg corresponds to an expected 0.2285/4 or approximately `5.71% positive difference` in probability of approve personal loan.

#- The standard deviation of `Mortgage` to the accepting personal loan is 101.71. The logistic regression coefficients corresponding to 1-standard-deviation differences are 0.0006 ∗  101.71 = 0.0610 (Beta5* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Mortgage corresponds to an expected 0.0610/4 or approximately `1.52% positive difference` in probability of approve personal loan. Remind that, the 1 unit in `Mortgage` is equal to `1 thousand dollar`.

#- The standard deviation of `Education` to the accepting personal loan is 0.84. The logistic regression coefficients corresponding to 1-standard-deviation differences are 1.6777 ∗  0.84 = 1.4092 (Beta4* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Education corresponds to an expected 1.4092/4 or approximately `35.23% positive difference` in probability of approve personal loan.

#- The standard deviation of `Family` to the accepting personal loan is 1.15. The logistic regression coefficients corresponding to 1-standard-deviation differences are 0.6869 ∗  1.15 = 0.7899 (Beta4* std deviasi). Dividing by 4 ( rule of 4) yields the quick summary estimate that a difference of 1 standard deviation in Family corresponds to an expected 1.4092/4 or approximately `19.74% positive difference` in probability of approve personal loan.

#-  Urutan probabilitas approving personal loan dari paling tinggi ke paling rendah:

#$$ Income > Education > Family > Experience > Credit Card Average Spending > Mortgage > Age $$

#### Evaluation

##### Log score for null model

#Consider we're just guessing the predicted probability of approving personal loan by flippling a coin (p=0.5), The log score is:

# predicted outcome (p)
# jika untuk pindah sumur memiliki probability 0.5
#mengunakan null model, artinya random saja apakah seseorang akan pindah sumur atau tidak
prob = 0.5

# true outcome
personalLoan = bank_loan["PersonalLoan"].copy()

logscore_null_model = np.sum(personalLoan * np.log(prob) + (1 - personalLoan) * np.log(1 - prob))
logscore_null_model
#logscore_null_model = (probability pindah sumur * 0.5) + (probability tidak pindah sumur * 1-0.5)

#Extract log score in DataFrame for more convenient comparison
np.sum(personalLoan * np.log(prob))

np.sum(1 - personalLoan) * np.log(1 - prob)

#Which the null log score is $-332.710 \log(0.5) - 3,133.025 \log(0.5) = -3,465$

#dataframe logscore 1-0.5

logscore = pd.DataFrame(data = logscore_null_model, columns = ["log_score"], index = ["null_model"])
logscore

#### Log score for baseline model

#Consider we guess the predicted probability of personalLoan by guessing using proportion of personal loan in our data

# the proportion of the "personalLoan" menggunakan baseline model
#dimana di set jika seseorang pindah sumur adalah 0.5
prob = round(np.sum(personalLoan)/len(personalLoan), 2) #len(personalLoan = 3020
prob
#The fact that 10% of the customer in the data approve personalLoan and 90% disapprove personalLoan, we can assign 10% to P(personnal loan = 1) for each respondent

# number of respondents who personalLoan the well
np.sum(personalLoan)

# number of respondents who not personalLoan the well
np.sum(1 - personalLoan)

#logscore_baseline_model - 1737*log0.58 + 1283*log0.42

logscore_baseline_model = np.sum(personalLoan * np.log(prob) + (1 - personalLoan) * np.log(1 - prob))
logscore_baseline_model

#Which improves the log score to $480 \log(0.1) + 4,520 \log(0.9) = -1,581$
logscore.loc["baseline_model","log_score"] = logscore_baseline_model
logscore

#Nilai logscore semakin besar maka, model logsitic regression akan semakin bagus modelnya.

#### Log score for logistic model

#Let's see if our fit further improves when we include predictors variables to predict personal loan while using Predicted probability values from statsmodel or logistic regression model to the model with probability guessing from the data yields:

# predicted probabilitas dari 3020 individu untuk personalLoan
pred_prob_personalLoan = model_personalLoan.predict(bank_loan[['Age','Experience','Income','CCAvg','Mortgage','Education','Family']])
pred_prob_personalLoan

#- Calculate log score manually
#kemudian tambahkan ke dataframe
logscore_logreg_model1 = np.sum(personalLoan * np.log(pred_prob_personalLoan) + (1 - personalLoan) * np.log(1-pred_prob_personalLoan))
logscore_logreg_model1

#- You can use llf results too under statsmodels model results
np.sum(personalLoan * np.log(pred_prob_personalLoan))

np.sum((1 - personalLoan) * np.log(1-pred_prob_personalLoan))

#Which improves the log score to $-469.25 \log - 260,87 \log = -730.12$#
#perbandingan null_model, baseline_model, dan logreg_model

logscore.loc["logreg_model1","log_score"] = logscore_logreg_model1
logscore

#Nilai logscore semakin besar maka, model logsitic regression akan semakin bagus modelnya. atau jika bernilai negatif, maka nilai mendekati 0 semakin bagus model logistic regresi nya.

#- The log score for the new model is -730.129

##### Difference of Log score

#Comparison our logistic regression model to the model with probability guessing from the data yields

#perbedaan dari logreg_model dengan null_model

diff_logscore = logscore["log_score"]["logreg_model1"] - logscore["log_score"]["null_model"]
diff_logscore

#- Including full predictor in the model clearly improves the predictive accuracy than just guessing the probability of personalLoan from the proportion of personal loan in the data.
#- The improvement in predictive probabilities (2735.60) describes how much more accurately we are predicting the probabilities.

##### Log Score - LOO Cross Validation

#Now, to have more generalized performance to, we calculate log score use LOO (Leave-One-Out) cross validation

#Helper Function

# model untuk logistic regresi LOO cross validation

from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold

def loo_split(data):
    """
    Function to combine estimated coefficients and standard error in one DataFrame
    :param data: <pandas DataFrame> 
    :param n_fold: <int> Number of fold in k-fold CV
    :return fold_train: <pandas DataFrame> Training Data
    :return fold_test: <pandas DataFrame> Testing Data
    """
    loo = LeaveOneOut()
    fold_train = {}
    fold_test = {}

    for i, (train, test) in enumerate(loo.split(data)):
        fold_train[i] = data.iloc[train].copy()
        fold_test[i] = data.iloc[test].copy()
        
    return (fold_train, fold_test)

def kfold_split(data, n_fold):
    """
    Function to combine estimated coefficients and standard error in one DataFrame
    :param data: <pandas DataFrame> 
    :param n_fold: <int> Number of fold in k-fold CV
    :return fold_train: <pandas DataFrame> Training Data
    :return fold_test: <pandas DataFrame> Testing Data
    """
    kfold = KFold(n_splits = n_fold, 
                  shuffle = True, 
                  random_state=123)
    fold_train = {}
    fold_test = {}

    for i, (train, test) in enumerate(kfold.split(data)):
        print(f"fold {i+1}, train data rows: {len(train)}, test data rows: {len(test)}")
        fold_train[i] = data.iloc[train].copy()
        fold_test[i] = data.iloc[test].copy()
        
    return (fold_train, fold_test)

#Data Splitting

fold_train, fold_test = loo_split(data = bank_loan)

fold_train[0].head()

fold_test[0]

#As LOO will leave one observation out as test data, so in each folds, we have 4999 observations as train data, and 1 observations as test data

#- Fit n times models and use it to predict 1 observation as test data
#- It may takes time, since loo create n times model 

# create empty list to store predicted probabilities
pred_list = []

# n data iteration to perform loo
for i in range(len(bank_loan)):

    # build model for each train fold
    model = smf.logit('PersonalLoan ~ Age + Experience + Income + Education + Family ', data = fold_train[i] )
    fit = model.fit(disp=0)
    
    # model prediction for each test fold
    pred = fit.predict(fold_test[i][['Age','Experience','Income', 'Education','Family']])
    
    # store the probability prediction to list
    pred_list.append(pred)
    
#Make new dataframe to save the predicted probabilities and y true of test data
prediction = pd.DataFrame(data = np.array(pred_list), columns=["test_pred_prob"])
prediction["y"] = [fold_test[i]["PersonalLoan"].values[0] for i in range(len(bank_loan))]
prediction.head()\

# Compute log score for each test data
prediction["log_score"] = prediction["y"] * np.log(prediction["test_pred_prob"]) \
                            + (1-prediction["y"]) * np.log(1 - prediction["test_pred_prob"])

prediction.head()

#- Cross validation score compute the average log score that resulted log score for 1 observation (since the test data only have 1 observation)

prediction["log_score"].mean()

#- We would like to now the log score of all observations to do comparison of other computed log score, thus we need compute the log score for all observation
#- We can use the cross validation score times the size of test data

prediction["log_score"].mean()*len(bank_loan)

#- or compute the log score manually

#sum logscore

logscore_logreg_model_loo = np.sum(prediction["y"] * np.log(prediction["test_pred_prob"]) \
                                   + (1 - prediction["y"]) * np.log(1-prediction["test_pred_prob"]))
logscore_logreg_model_loo

#- It will have the same result
# Store to logscore dataframe 

logscore.loc["logreg_model_nocc&mrtg_loo","log_score"] = logscore_logreg_model_loo
logscore

#- The LOO estimated  the log score of model with no CCAvg & Mortgage $\approx$ -742 is 12 lower than log score of -730 computed before; 
#- This difference is about what we would expect, given that the fitted model has 5 parameters or degrees of freedom compared to previous model that has 7 predictor that including CCAvg & mortgage.

### Fit Logistic Regression - Add Interaction

#Next, try to add interaction and see the computed logscore

#model logistic regerssion dengan tambah model interaksi.

# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Age + Experience + Income + CCAvg + Mortgage + Education + Family + CCAvg:Mortgage", bank_loan)

# Fit the model
model_personalLoan_i1 = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_personalLoan_coef_i1 = print_coef_std_err(model_personalLoan_i1)
results_personalLoan_coef_i1

#### Coefficient Interpretation 

#- Now we read the result of the model, and interpret each coefficient
results_personalLoan_coef_i1

#$$P(\text{PersonalLoan}) = \text{logit}^{-1}(-12.82 -0.03\text{Age} + 0.051\text{Experience} + 0.053\text{Income} + 0.145\text{CCAvg} + 0.0012\text{Mortgage} + 1.685\text{Education} + 0.6855\text{Family} - 0.00017 \text{CCAvg} \times \text{Mortgage}) $$

#To understand the numbers in the table, we use the following tricks:
#rata2 dari CCAvg
bank_loan["CCAvg"].mean()

#rata2 dari Mortgage
bank_loan["Mortgage"].mean()

#- Evaluating predictions and interactions at the mean of the data, which have average values of 1.937 for CCAvg and 56.49 for Mortgage (that is, a mean `CCAvg of 1.937 thousand dollar` to the approving personal loan well, and a mean `mortgage level of 56.49 thousand dollar` among the personal loan).
#- Consider other predictor beside CCAvg and Mortgage are all have the same intepretation as before, since in this model have interaction of CCAvg and mortgage, for these 2 variable have a new intepretationm using dividing by 4 to get approximate predictive differences on the probability scale.
   
#1. Coefficient for CCAvg
#    - This corresponds to comparing two bank_loan that differ by 1 in CCAvg, if *the Mortgage level is 0* for both bank_loan. Once again, we should not try to interpret this. 
#    - Instead, we can look at the average value, Mortgage = 56.49, where CCAvg has a coefficient of 0.1459 − 0.00017 ∗ 56.49 = 8.2322 on the logit scale. 
#    - To quickly interpret this on the probability scale, we divide by 4: 8.2322/4 = 2.0580. Thus, at the mean level of Mortgage in the data, each unit increasing of CCAvg corresponds to an approximate `205.80% positive difference` in probability of accepting personal loan.
    
#2. Coefficient for Mortgage
#    - This corresponds to comparing two bank_loan that differ by 1 in Mortgage, if *the CCAvg to the nearest safe well is 0* for both. 
#    - Again, this is not so interpretable, so instead we evaluate the comparison at the average value for CCAvg = 1.9379, where Mortgage has a coefficient of 0.0012 - 0.00017 ∗ 1.9379 = 0.0019 on the logit scale. 
#    - To quickly interpret this on the probability scale, we divide by 4: 0.0019/4 = 0.00049. Thus, at the mean level of CCAvg in the data, each additional unit of increasing in Mortgage corresponds to an approximate `0.049% positive difference` in probability of personal loan.
    
#4. Coefficient for the interaction term. This can be interpreted in two ways.
#    - Each additional unit increasing of  Mortgage, the value -0.00017 is added to the coefficient for CCAvg. The coefficient for CCAvg is 8.2322 at the average level of Mortgage, thus, we can understand the interaction as saying that `the importance of CCAvg as a predictor decrease for customer with higher existing Mortgage levels`.
#    - For each additional unit increasing of CCAvg to the accepting personal loan, the value -0.00017 is added to the coefficient for Mortgage. The coefficient for Mortage is 0.0019 `at the average CCAvg to accepting personal loan, thus the importance of Mortgage as a predictor decreases for approving personal loan for customer with higher CCAvg level`.
    
#### Log Score - LOO
# Split data
fold_train, fold_test = loo_split(data = bank_loan)

# create empty list to store predicted probabilities
pred_list = []

# LOO CV
for i in range(len(bank_loan)):

    # build model for each train fold
    model = smf.logit("PersonalLoan ~ Age + Experience + Income + CCAvg + Mortgage + Education + Family + CCAvg:Mortgage", data = fold_train[i] )
    fit = model.fit(disp=0)
    
    # model prediction for each test fold
    pred = fit.predict(fold_test[i][['Age','Experience','Income','CCAvg','Mortgage','Education','Family']])
    
    # store the probability prediction to list
    pred_list.append(pred)

# Compute LOO Log Score

prediction = pd.DataFrame(data = np.array(pred_list), columns=["test_pred_prob"])

prediction["y"] = [fold_test[i]["PersonalLoan"].values[0] for i in range(len(bank_loan))]
prediction["log_score"] = prediction["y"] * np.log(prediction["test_pred_prob"]) + (1-prediction["y"]) * np.log(1 - prediction["test_pred_prob"])
logscore_logreg_interaction_loo = prediction["log_score"].mean()*len(bank_loan)

# Store the log score
logscore.loc["logreg_interaction_loo","log_score"] = logscore_logreg_interaction_loo
logscore

## log score difference the current model, interaction model with previous model (with no iteraction)

logscore["log_score"]["logreg_interaction_loo"] - logscore["log_score"]["logreg_model_nocc&mrtg_loo"]

### Centering
#centering dengan mengkurangkan data nta dengan rata2

bank_loan["c_Age"] = bank_loan["Age"]- bank_loan["Age"].mean()
bank_loan["c_Experience"] = bank_loan["Experience"]- bank_loan["Experience"].mean()
bank_loan["c_Income"] = bank_loan["Income"]- bank_loan["Income"].mean()
bank_loan["c_CCAvg"] = bank_loan["CCAvg"]- bank_loan["CCAvg"].mean()
bank_loan["c_Mortgage"] = bank_loan["Mortgage"]- bank_loan["Mortgage"].mean()
bank_loan["c_Education"] = bank_loan["Education"]- bank_loan["Education"].mean()
bank_loan["c_Family"] = bank_loan["Family"]- bank_loan["Family"].mean()

### Fit Logistic Regression - Add Full Predictor
# Create Logit model object
logit_model = smf.logit("PersonalLoan ~ Age + Experience + Income + CCAvg + Mortgage + Education + Family", bank_loan)

# Fit the model
model_PersonalLoan_edu = logit_model.fit()

# Extract the results (Coefficient and Standard Error) to DataFrame
results_PersonalLoan_coef_edu = print_coef_std_err(model_PersonalLoan_edu)
results_PersonalLoan_coef_edu

#### Coefficient Interpretation 

#- Now we read the result of the model, and interpret each coefficient
results_PersonalLoan_coef_edu

#**Intercept**

#- This value represent inverse logit of -12.80 is the estimated probability of Personal Loan, if all centering predictor variable = 0

expit(-12.80)

#logit inverse probabilitias untuk intercept

#- That is, if all predictors are at their averages in the data, the probability of approving personal loan is 2.7607649501930464e-04%.
   
#**Coefficient for c_Age**
#- To quickly interpret this on the probability scale, we divide by 4: -0.0375/4 = -0.0093. Thus, at the mean level of arsenic in the data, each 100 meters of distance corresponds to an approximate 0.93% negative difference in probability of approving personal loan.

#**Coefficient for c_Experience**
#- To quickly interpret this on the probability scale, we divide by 4: 0.0499/4 = 0.0124. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 1.24% positive difference in probability of personal loan.

#**Coefficient for c_Income**
#- To quickly interpret this on the probability scale, we divide by 4: 0.0540/4 = 0.0135. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 1.35% positive difference in probability of personal loan.

#**Coefficient for c_CCAvg**
#- To quickly interpret this on the probability scale, we divide by 4: 0.1306/4 = 0.0326. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 3.26% positive difference in probability of personal loan.

#**Coefficient for c_Mortgage**
#- To quickly interpret this on the probability scale, we divide by 4: 0.00065/4 = 0.00016. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 0.016% positive difference in probability of personal loan.

#**Coefficient for c_Education**
#- To quickly interpret this on the probability scale, we divide by 4: 1.677/4 = 0.4194. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 41.94% positive difference in probability of personal loan.

#**Coefficient for c_Family**
#- To quickly interpret this on the probability scale, we divide by 4: 0.6869/4 = 0.1717. Thus, at the mean level of distance in the data, each additional unit of arsenic corresponds to an approximate 17.17% positive difference in probability of personal loan.

#**Coefficient for Education**
#- Respondents with higher education are more likely to say they would switch bank_loan: the crude estimated difference is 1.677/4 = 0.4194, or a 4% positive difference in personal loan probability when comparing households that differ by 4 years of education. 
#- The coefficient for education makes sense and is estimated fairly precisely—its standard error is much lower than the coefficient estimate.

#### Log Score - LOO
# Split data
fold_train, fold_test = loo_split(data = bank_loan)

# create empty list to store predicted probabilities
pred_list = []

# LOO CV
for i in range(len(bank_loan)):

    # build model for each train fold
    model = smf.logit("PersonalLoan ~ c_Age + c_Experience + c_Income + c_CCAvg + c_Mortgage + c_Education + c_Family", data = fold_train[i] ) #namun lebih baik menggunakan model yang sudah di centered
    fit = model.fit(disp=0)
    
    # model prediction for each test fold
    pred = fit.predict(fold_test[i][['c_Age','c_Experience','c_Income','c_CCAvg','c_Mortgage','c_Education','c_Family']])
    
    # store the probability prediction to list
    pred_list.append(pred)

# Compute LOO Log Score
prediction = pd.DataFrame(data = np.array(pred_list), columns=["test_pred_prob"])

prediction["y"] = [fold_test[i]["PersonalLoan"].values[0] for i in range(len(bank_loan))]
prediction["log_score"] = prediction["y"] * np.log(prediction["test_pred_prob"]) + (1-prediction["y"]) * np.log(1 - prediction["test_pred_prob"])
# Log Score for all data
logscore_logreg_model_centering_loo = prediction["log_score"].mean()*len(bank_loan)

# Store the log score
logscore.loc["logreg_model_centering_loo","log_score"] = logscore_logreg_model_centering_loo
logscore

logscore["log_score"]["logreg_model_centering_loo"] - logscore["log_score"]["logreg_interaction_loo"]
#Adding centering the variable improves predictive log score (+0.6354) from the log score of interaction.

#Urutan model logistic regression dari yang terbaik adalah sebagai berikut:

#$$ logregmodel1 > logregmodelcenteringloo > logreginteractionloo > logregmodelnoccandmrtgloo > baselinemodel > nullmodel $$

