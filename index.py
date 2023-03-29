#
# - - - - - NOTES - - - - - 



# - - - - - IMPORTS - - - - - 
import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool, cv
from sklearn import linear_model, metrics, model_selection, preprocessing, tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sns.set_style('whitegrid')


# - - - - - FUNCTIONS - - - - - 
#def ():
#    pass

def sns_plot_function(plot_type, data, column_x='', column_y=''):
#sns_plot_function(tipo, dataframe, eixo_x, eixo_y, kde<bool>, hue)
    #print(f'Quant. valores únicos: {data[column_x].nunique()}')
    #print(f'Quais os valores únicos: {data[column_x].unique()}')
    #print(f'Quant. valores nulos: {data[column_x].isnull().sum()}')
    #print(f'Quant. por opção:\n{data[column_x].value_counts()}')
    if plot_type == 'countplot':
        sns.countplot(data=data, x=column_x, hue=column_x)
    elif plot_type == 'displot':
        sns.displot(data[column_x], kde=True)
    elif plot_type == 'jointplot':
        sns.jointplot(data=data, x=column_x, y=column_y, hue=column_y)
    elif plot_type == 'pairplot':
        sns.pairplot(data=data)
    elif plot_type == 'lmplot':
        sns.lmplot(data=data, x=column_x, y=column_y)

def accuracy_func(algo, x_train, y_train, cross_val):
    model = algo.fit(x_train, y_train)
    accuracy = round(model.score(x_train, y_train) * 100, 2)
    train_predict = model_selection.cross_val_predict(algo, x_train, y_train, cv = cross_val, n_jobs=-1)
    cross_val_accuracy = round(metrics.accuracy_score(y_train, train_predict) * 100, 2)
    return accuracy, cross_val_accuracy


# - - - - - CLEANING / EDA - - - - - 
#df_train.info() #df_train.head(3) #df_train.tail() #.columns
#df_train['Avatar'].unique() #.isnull().sum() #df_train['Avatar'].value_counts()
#df_train.describe()

#dataset treino
df_train = pd.read_csv('./ecommerce-customers')
#500 entradas
#valores nulos: 0
#'24645 Valerie Unions Suite 582\r\nCobbborough, DC 99414-7564'

#Avg. Session Length: tempo médio das sessões de consultoria na loja
#Time on App: tempo médio gasto no app (min)
#Time on Website: tempo médio gasto no site (min)
#Length of Membership: tempo que o cliente é membro. (anos)
#média: 3.5 anos; menor: 0.269901 x 12 = 3meses8dias; maior: 7 anos

#Yearly Amount Spent: montante gasto por ano ($)
#média : $499.3; menor: $256.6; maior: $765.5
# USD 1 = BRL 5.29 (15/03/23)

#dataset teste
#df_test = pd.read_csv('./test.csv')
# entradas
#valores nulos: ; ;

sns_plot_function('lmplot', df_train, 'Length of Membership', 'Yearly Amount Spent')
#sns.jointplot(df_train, x='Yearly Amount Spent', y='Time on App', kind='hex');
#correlação entre 'Length of Membership' e 'Yearly Amount Spent'

#df_train['Address'][2]
#df_train['Address'][7].split('\r\n')[1]
#df_train['Address'][3].split('\r\n')[1].split(',')[0] #cidade
#df_train['Address'][3].split('\r\n')[0].split(' ', 1)[1] #logradouro
#[df_train['Address'][n].split('\r\n')[1].split(',')[0] for n in df_train.index]

# - - - - - TRAIN / TUNING / VALIDATE / TEST - - - - - 
#treino
y = df_train['Yearly Amount Spent']
x= df_train[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(x_train, y_train)

#print('Coeficientes: \n', lm.coef_) # [25.98154972 38.59015875  0.19040528 61.27909654]

#tuning

#cross validation

#teste
predictions = lm.predict(x_test)
plt.scatter(y_test, predictions);
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#print('MAE:', metrics.mean_absolute_error(y_test, predictions)) #7.228148653430811
#print('MSE:', metrics.mean_squared_error(y_test, predictions)) #79.81305165097385
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) #8.9338150669786

sns.distplot((y_test - predictions), bins=50);

# - - - - - END - - - - - 
#
