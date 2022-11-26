import yfinance as yf
import datetime as date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statistics import mean

today= date.datetime.today()
end= today - date.timedelta(365)
S1 = yf.download("F", end, today )
returns = pd.DataFrame.pct_change(S1.Close)
returns['trans_returns']= np.arcsinh(returns)
returns=pd.DataFrame(returns)
# plt.hist(returns['trans_returns'])
# plt.show()

x1= yf.download("F", end, today)
x2=yf.download('A', end, today)
x3= yf.download("TSLA", end, today)
D2= pd.DataFrame(pd.concat([x1.Close, x2.Close, x3.Close], axis=1))
"""
   This function combines the tests for stationarity. Adopted from function given by Gary Cornwall

   1.) Visual Inspection 
   2.) Standard Deviation Check
   3.) Augmented Dicky Fuller Test
   4.) P/ACF Inspection
   
   """

#making the stationarity function
def Intord(Series):
    if isinstance(Series, pd.Series):
        #calculations
        diff = np.diff(Series)
        diff_2= np.diff(diff)
        x= round(np.std(Series),2)
        x_2= round(np.std(diff),2)
        x_3 = round(np.std(diff_2),2)
        ADF = adfuller(Series)
        PACF = sm.graphics.tsa.pacf(Series, nlags=int(len(Series)*.2))
        ACF = sm.graphics.tsa.acf(Series, nlags=int(len(Series)*.2))
        lags= range(0,int(len(Series)*.2) +1)
        day = date.datetime.today()
        name= Series.name

        #figures
        figure = plt.figure()
        figure.suptitle(f"Stationarity Tests: Run on {day}", fontsize=20)

        ax_1= figure.add_subplot(321)
        ax_2= figure.add_subplot(323)
        ax_3= figure.add_subplot(325)
        ax_4= figure.add_subplot(324)
        ax_5= figure.add_subplot(322)
        ax_6= figure.add_subplot(326)
        ax_1.set_autoscale_on(True)


        ax_1.plot(Series)
        ax_1.text(0.04,0.05,f"Standard deviation: {x}",bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5}, transform=ax_1.transAxes)
        ax_1.set_title("Level Plot")
        ax_1.axhline(y=int(mean(Series)), xmin=0, xmax=len(Series))
        ax_2.plot(diff)
        ax_2.set_title("Differenced Plot")
        ax_2.axhline(y=int(mean(diff)), xmin=0, xmax=len(diff))
        ax_2.text(0,-1.9,f"Standard deviation: {x_2}",bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5})
        ax_3.plot(diff_2)
        ax_3.set_title("Second Difference Plot")
        ax_3.axhline(y=int(mean(diff_2)), xmin=0, xmax=len(diff_2))
        ax_3.text(0,-3.1,f"Standard deviation: {x_3}", bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5})

        ax_4.bar(lags,PACF, width=.2)
        ax_4.set_title("Partial Autocorrelation Function")
        ax_4.axhline(y=0.0, xmin=0, xmax=len(PACF))
        ax_5.bar(lags,ACF, width=.2)
        ax_5.set_title("Autocorrelation Function")
        ax_6.text(.5, 0.6, 'Augmented Dicky- Fuller Test',
            verticalalignment='center', horizontalalignment='center',
            transform=ax_6.transAxes,
            color='black', fontsize=15)
        ax_6.text(.5, 0.4, f'Test Statistic {round(ADF[0],2)}, P-Value {round(ADF[1],2)}',
            verticalalignment='center', horizontalalignment='center',
            transform=ax_6.transAxes,
            color='black', fontsize=12)
        ax_6.set_xticks([])
        ax_6.set_yticks([])
        plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
        plt.show()

    if isinstance(Series, pd.DataFrame):
        for column in range(0,len(Series)):
            series=pd.Series(Series.iloc[:,column])
            #calculations
            diff = np.diff(series)
            diff_2= np.diff(diff)
            x= round(np.std(series),2)
            x_2= round(np.std(diff),2)
            x_3 = round(np.std(diff_2),2)
            ADF = adfuller(series)
            PACF = sm.graphics.tsa.pacf(series, nlags=int(len(series)*.2))
            ACF = sm.graphics.tsa.acf(series, nlags=int(len(series)*.2))
            lags= range(0,int(len(series)*.2) +1)
            day = date.datetime.today()
            name= series.name

            #figures
            figure = plt.figure()
            figure.suptitle(f"Stationarity Tests: Run on {day}", fontsize=20)
            ax_1= figure.add_subplot(321)
            ax_2= figure.add_subplot(323)
            ax_3= figure.add_subplot(325)
            ax_4= figure.add_subplot(324)
            ax_5= figure.add_subplot(322)
            ax_6= figure.add_subplot(326)
            ax_1.set_autoscale_on(True)


            ax_1.plot(series)
            ax_1.text(0.04,0.05,f"Standard deviation: {x}",bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5}, transform=ax_1.transAxes)
            ax_1.set_title("Level Plot")
            ax_1.axhline(y=int(mean(series)), xmin=0, xmax=len(series))
            ax_2.plot(diff)
            ax_2.set_title("Differenced Plot")
            ax_2.axhline(y=int(mean(diff)), xmin=0, xmax=len(diff))
            ax_2.text(0,-1.9,f"Standard deviation: {x_2}",bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5})
            ax_3.plot(diff_2)
            ax_3.set_title("Second Difference Plot")
            ax_3.axhline(y=int(mean(diff_2)), xmin=0, xmax=len(diff_2))
            ax_3.text(0,-3.1,f"Standard deviation: {x_3}", bbox={'facecolor': 'lightgrey', 'alpha': 1, 'pad': 2.5})

            ax_4.bar(lags,PACF, width=.2)
            ax_4.set_title("Partial Autocorrelation Function")
            ax_4.axhline(y=0.0, xmin=0, xmax=len(PACF))
            ax_5.bar(lags,ACF, width=.2)
            ax_5.set_title("Autocorrelation Function")
            ax_6.text(.5, 0.6, 'Augmented Dicky- Fuller Test',
                verticalalignment='center', horizontalalignment='center',
                transform=ax_6.transAxes,
                color='black', fontsize=15)
            ax_6.text(.5, 0.4, f'Test Statistic {round(ADF[0],2)}, P-Value {round(ADF[1],2)}',
                verticalalignment='center', horizontalalignment='center',
                transform=ax_6.transAxes,
                color='black', fontsize=12)
            ax_6.set_xticks([])
            ax_6.set_yticks([])
            plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
            plt.show()
    else:
        raise TypeError("This is not a Pandas DataFrame or Series")



