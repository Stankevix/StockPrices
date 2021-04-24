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
import itertools
import warnings


import statsmodels.tsa.stattools as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.seasonal import seasonal_decompose

STEPS = 10
TARGET = 'Último'
EXOG = ['Covid','CriticalCovid']


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
    df.loc[1430:1640,'CriticalCovid'] = 1
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


#funcao para achar os parametros otimos do sarimax - força bruta
def opt_parameters(ts,ts_test,pdq, pdqs):
    ts_exog = ts[EXOG]
    ts_test_exog = ts_test[EXOG]
    
    ts = ts[TARGET] 
    ts_test = ts_test[TARGET]
    
    
    best_param = []
    for param in pdq:
        for params in pdqs:
            try:
                mod = sarimax.SARIMAX(ts,
                                      order=param,
                                      seasonal_order=params
                                      ) #exog = ts_exog
    
                model_fit = mod.fit(disp=False)
                
                pred_y = model_fit.get_forecast(steps=STEPS)#,
                                
                rmse = np.sqrt(mean_squared_error(ts_test,pred_y.predicted_mean))
                
                
                best_param.append([param, params, model_fit.bic,model_fit.aic, rmse])
                print('SARIMAX {} x {}12 : BIC Calculated ={}'.format(param, params, model_fit.bic))
            except:
                continue
    
    best_param_df = pd.DataFrame(best_param, columns=['pdq', 'pdqs', 'bic','aic','rmse'])
    best_param_df = best_param_df.sort_values(by=['rmse'],ascending=True)[0:5]
    
    return best_param_df



def auto_sarimax(n, m, ts,ts_test):
    
    p = q = range(n, m)
    d = range(0,2)
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    opt_param = opt_parameters(ts,ts_test, pdq, pdqs)
    
    return opt_param


def get_decompose_analysis(df, period):
    result = seasonal_decompose(itausa['Último'].values,model='additive',period=period)
    result.plot()


# In[3]: Leitura dos dados

itausa_raw = pd.read_csv('dataset_itsa4.csv',',')


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
df_itausa_train = itausa[900:1530]
df_itausa_test = itausa[1530:1540]

# In[9]:
    
model_params = auto_sarimax(0, 3, df_itausa_train,df_itausa_test)

# In[9]:
df_itausa_train['Último'].plot()
df_itausa_test['Último'].plot()


# In[9]:

    
# In[11]: Predição valores historicos - Teste

model = sarimax.SARIMAX(df_itausa_train['Último'],
                        order=(2,1,2),
                        seasonal_order=(2,0,1,12),
                        exog=df_itausa_train[EXOG]) #exog=df_itausa_train[EXOG]
model_fit = model.fit(disp=False)

pred_y = model_fit.get_forecast(steps=STEPS,
                                exog=df_itausa_test[EXOG]) #,exog=df_itausa_test[EXOG]

itausa_pred = pred_y.predicted_mean
itausa_conf = pred_y.conf_int()

res_acc = forecast_accuracy(df_itausa_test['Último'],itausa_pred)


# In[11]: Pred

train_pred_y = model_fit.get_prediction()
train_itausa_pred = train_pred_y.predicted_mean
train_itausa_conf = train_pred_y.conf_int()    
   

train_res_acc = forecast_accuracy(df_itausa_train['Último'],train_itausa_pred)