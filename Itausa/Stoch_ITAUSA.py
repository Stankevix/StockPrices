#!/usr/bin/env python
# coding: utf-8

# # Predição de ações Itausa (ITSA4)

# ## Historia ITAUSA

# Itausa Investimentos Itau SA é uma empresa sediada no Brasil e tem como atividade principal o setor bancário. As atividades da Companhia estão divididas em dois segmentos de negócios: Financeiro e Industrial. 
# 
# A divisão Financeiral concentra-se na gestão do Itau Unibanco Holding SA, uma instituição bancária que oferece produtos e serviços financeiros, como empréstimos, cartões de crédito, contas correntes, apólices de seguros, ferramentas de investimento, corretagem de valores mobiliários, consultoria de tesouraria e investimentos para clientes individuais e empresas. 
# 
# A divisão Industrial é responsável pela operação da Itautec SA, que fabrica equipamentos de automação comercial e bancária, além de prestar serviços de tecnologia da informação (TI); Duratex SA, que produz painéis de madeira, louças sanitárias e metais sanitários, e Alpargatas, que produz calçados sob as marcas Juntas, Havaianas e Dupe, entre outros.

# ## Funções
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_style('darkgrid')

warnings.simplefilter("ignore")

# In[2]: Funções

def adf_test(dataset, log_test = False):
    ds = dataset
    
    if log_test:
        ds = np.log(ds)
        ds.dropna(inplace=True)
    
    alpha = 0.05
    
    result = tsa.adfuller(ds)
    print('Augmented Dickey-Fuller Test')
    print('test statistic: %.10f' % result[0])
    print('p-value: %.10f' % result[1])
    print('critical values')
    
    for key, value in result[4].items():
        print('\t%s: %.10f' % (key, value))
        
    if result[1] < alpha:  #valor de alpha é 0.05 ou 5 %
        print("Rejeitamos a Hipotese Nula\n")
        return 1
    else:
        print("Aceitamos a Hipotese Nula\n")
        return 0

def get_stationary(df):

    if(adf_test(df['Último'], True) == 0):
        n_diff_dataset = pd.DataFrame(data=np.diff(np.array(df['Último'])))
        n_diff_dataset.columns = ['Último']
        adf_test(n_diff_dataset['Último'],False)
        return 
    return

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse})


def create_features(df):
    df['Data'] = pd.to_datetime(df['Data']).dt.date
    df['Mes'] = pd.to_datetime(df['Data']).dt.month
    df['Quadrimestre'] = pd.to_datetime(df['Data']).dt.quarter
    df['Dia_da_Semana'] = pd.to_datetime(df['Data']).dt.dayofweek
    
    df = df.set_index('Data').asfreq('d')
    
    df = df.interpolate(method='linear')
    
    return df

def get_critical_covid(df):
    df['CriticalCovid'] = 0
    df.loc[1430:1640,'CriticalCovid'] = -1
    return df


def get_order_diff(df,nLags): 
    
    fig, axes = plt.subplots(3, 2,figsize=(15,12))

    axes[0, 0].plot(df['Último']); axes[0, 0].set_title('Original Series')
    plot_acf(df['Último'], ax=axes[0, 1],lags=nLags)
    
    # 1st Differencing
    axes[1, 0].plot(df['Último'].diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df['Último'].diff().dropna(), ax=axes[1, 1])
    
    # 2nd Differencing
    axes[2, 0].plot(df['Último'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df['Último'].diff().diff().dropna(), ax=axes[2, 1])
    
    plt.show()

def get_trend_plots(df):
    
    f, ax = plt.subplots(figsize=(15, 5))

    sns.histplot(data=itausa, x=itausa['Último'],kde=True)
    ax.set(title="Histogram for ITSA4")
    plt.show()
    
    f, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x="Data", y="Último",label ='Ultima',data=itausa)
    sns.lineplot(x="Data", y="Máxima",label ='Maxima',data=itausa)
    sns.lineplot(x="Data", y="Mínima",label ='Minima',data=itausa)
    plt.suptitle("ITSA4 Values")
    plt.show()
    
    f, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x="Data", y="Vol.",label ='Volume Diario',data=itausa)
    plt.suptitle("ITSA4 Values")
    plt.show()
    
    f, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x="Data", y="Var%",label ='variation',data=itausa)
    plt.suptitle("ITSA4 Variation")
    plt.show()
    
    f, ax = plt.subplots(figsize=(15, 5))
    sns.lineplot(x="Data", y="Vol.",data=itausa)
    plt.suptitle("ITSA4 Volume")
    plt.show()


def get_decompose_analysis(df, period):
    result = seasonal_decompose(itausa['Último'].values,model='additive',period=period)
    result.plot()


# In[3]: Leitura dos dados

itausa_raw = pd.read_csv('dataset_itsa4.csv',';')


# In[4]: Criação de novas Features

itausa = create_features(itausa_raw)
itausa = get_critical_covid(itausa)

corr = itausa.corr()

# In[5]: Plotar graficos de tendencia
get_trend_plots(itausa)

# In[6]: Avaliar estacionariedade da serie
    
get_stationary(itausa)

# In[7]: Avaliar niveis de lags / diferenciacao
nLags = 150
get_order_diff(itausa,nLags)


# In[8]: Analise Decomposição

get_decompose_analysis(itausa, 30)

# In[9]:
df_itausa_train = itausa[635:1630]
df_itausa_test = itausa[1630:]



# In[9]:
df_itausa_train['Último'].plot()
df_itausa_test['Último'].plot()


# In[9]:
'''
def get_best_model(df_train, df_test, params):
    
    p_values = range(0,3)
    d_values = range(0,2)
    q_values = range(0,3)
    ps_values = range(0,1)
    ds_values = range(0,1)
    qs_values = range(0,1)
    
    lowest_RMSE = 999999
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for ps in ps_values:
                    for ds in ds_values:
                        for qs in qs_values:
                            iorder = (p,d,q)
                            iseasonal_order = (ps,ds,qs,1)
                            train,test = df_itausa_train['Último'], df_itausa_test['Último']
                            try:
                                model = sarimax.SARIMAX(train,order=iorder,seasonal_order=iseasonal_order,exog=df_itausa_train[params])
                                model_fit = model.fit(disp=False)
                                pred_y = model_fit.get_forecast(steps=194,exog=df_itausa_test[params]) 
        
                                RMSE = np.sqrt(mean_squared_error(test,pred_y.predicted_mean))
                                print('SARIMAX Order%s Seasonal_Order%s RMSE = %.2f'% (iorder,iseasonal_order,RMSE))
                                if(RMSE<lowest_RMSE):
                                    lowest_RMSE = RMSE
                                    best_order = iorder
                                    best_seasonal_order = iseasonal_order
                        except:
                            continue
    
    
    return model, model_fit,best_order, best_seasonal_order,lowest_RMSE
'''
        
# In[10]:

#get_best_model(df_train, df_test, params)  

    
# In[10]:
#params = ['Vol.','Covid','CriticalCovid']
params = ['Covid','CriticalCovid']
#params = ['Covid']

p_values = range(0,4)
d_values = range(0,3)
q_values = range(0,4)
ps_values = range(0,2)
ds_values = range(0,1)
qs_values = range(0,2)
lowest_RMSE = 999999

for p in p_values:
    for d in d_values:
        for q in q_values:
            for ps in ps_values:
                for ds in ds_values:
                    for qs in qs_values:
                        iorder = (p,d,q)
                        iseasonal_order = (ps,ds,qs,12)
                        train,test = df_itausa_train['Último'], df_itausa_test['Último']
                        try:
                            model = sarimax.SARIMAX(train,order=iorder,seasonal_order=iseasonal_order,exog=df_itausa_train[params])#,exog=df_itausa_train[params]
                            model_fit = model.fit(disp=False)
                            pred_y = model_fit.get_forecast(steps=90,exog=df_itausa_test[params][0:90]) #,exog=df_itausa_test[params]
    
                            RMSE = np.sqrt(mean_squared_error(test[0:90],pred_y.predicted_mean))
                            print('SARIMAX Order%s Seasonal_Order%s RMSE = %.2f'% (iorder,iseasonal_order,RMSE))
                            if(RMSE<lowest_RMSE):
                                lowest_RMSE = RMSE
                                best_order = iorder
                                best_seasonal_order = iseasonal_order
                        except:
                            continue
    
# In[11]: Predição valores historicos - Treinamento

model = sarimax.SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order,exog=df_itausa_train[params])
model_fit = model.fit(disp=False)

pred_y = model_fit.get_prediction(steps=995)
itausa_pred = pred_y.predicted_mean
itausa_conf = pred_y.conf_int()

res_acc = forecast_accuracy(itausa_pred, train)
print("Train",res_acc)

    
# In[11]: Predição valores historicos - Teste

model = sarimax.SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order,exog=df_itausa_train[params])
model_fit = model.fit(disp=False)

pred_y = model_fit.get_forecast(steps=5,exog=df_itausa_test[params][0:5])

itausa_pred = pred_y.predicted_mean
itausa_conf = pred_y.conf_int()

res_acc = forecast_accuracy(itausa_pred, test[0:5])
print("Test",res_acc)


# In[11]: Forecasting 3 dias
'''
df_itausa_train = itausa[1000:]
train,test = df_itausa_train['Último'], df_itausa_test['Último']
 
model = sarimax.SARIMAX(train,order=best_order,seasonal_order=best_seasonal_order,exog=df_itausa_train[params])
model_fit = model.fit(disp=False)

pred_y = model_fit.get_forecast(steps=3)
itausa_frc = pred_y.predicted_mean
itausa_frc_conf = pred_y.conf_int()    
'''    

