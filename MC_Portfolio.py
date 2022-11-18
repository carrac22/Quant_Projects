import datetime as dt
import yfinance as yf
import numpy as np
import scipy.optimize as sc
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime as dt

class Portfolio:
    def __init__(self, stocks, start, end, weights):
        self.weights = np.array(weights)
        self.startDate= start
        self.endDate= end
        self.stockData = yf.download(stocks, start=start, end=end)
        self.stockData = self.stockData['Close']
        self.returns = self.stockData.pct_change()
        self.meanReturns = self.returns.mean()
        self.covMatrix = self.returns.cov()
        self.Days = len(self.stockData)
        self.TotalDays= end - start
        self.names= self.stockData.columns
#this works
    def portfolioPerformance(self):
        WeightedReturns = np.sum(self.meanReturns*self.weights)*self.Days
        WeightedStd = np.sqrt(np.dot(self.weights.T,np.dot(self.covMatrix, self.weights)))*np.sqrt(self.Days)
        return "Weighted Returns:      "+ str(round(WeightedReturns*100,2)) +"%",  "Weighted Standard Deviation:      "+str(round(WeightedStd*100,2))+"%"
#this works
    def maxSR(self,  riskFreeRate = 0, constraintSet=(0,1)):
        "Minimize the negative SR, by altering the weights of the portfolio"
        numAssets = len(self.meanReturns)
        args = (self.meanReturns, self.covMatrix, self.Days, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))

        def negativeSR(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
            def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                WeightedReturns = np.sum(meanReturns*weights)*Days
                WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                return WeightedReturns,  WeightedStd

            pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Days)
            return - (pReturns - riskFreeRate)/pStd

        result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        allocations= result['x']
        dataf= pd.DataFrame(allocations, index=self.meanReturns.index , columns='Weights' )
        SharpeRatio= round(result['fun'], 2)
        return dataf
#this works
    def minVar(self, riskFreeRate = 0, constraintSet=(0,1)):
        "Minimize the negative SR, by altering the weights of the portfolio"
        numAssets = len(self.meanReturns)
        args = (self.meanReturns, self.covMatrix, self.Days, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))
        def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
            def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                WeightedReturns = np.sum(meanReturns*weights)*Days
                WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                return WeightedReturns,  WeightedStd
            return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
        result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result
#this works
    def efficientOptimization(self, returnTarget, constraintSet=(0,1)):
        numAssets = len(self.meanReturns)
        args= (self.meanReturns, self.covMatrix, self.Days)

        def portfolioReturns(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
            def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                WeightedReturns = np.sum(meanReturns*weights)*Days
                WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                return WeightedReturns,  WeightedStd
            return portfolioPerformance(weights, meanReturns, covMatrix, Days)[0]

        constraints= ({'type':'eq', 'fun': lambda x: portfolioReturns(x, self.meanReturns, self.covMatrix, self.Days)- returnTarget},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constraintSet
        bounds = tuple(bound for asset in range(numAssets))

        def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
            def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                WeightedReturns = np.sum(meanReturns*weights)*Days
                WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                return WeightedReturns,  WeightedStd
            return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
        EffOpt= sc.minimize(portfolioVariance, numAssets*[1./numAssets],args=args, method= 'SLSQP', bounds=bounds, constraints=constraints)
        return EffOpt
#this works
    def calculatedResults(self, riskFreeRate=0, constraintSet=(0,1)):
        """Read in Cov Matrix and other financial info and input max sharpe/ min volit/ and efficient frontier"""
        def maxSR(meanReturns, covMatrix, Days, riskFreeRate = 0, constraintSet=(0,1)):
            "Minimize the negative SR, by altering the weights of the portfolio"
            numAssets = len(meanReturns)
            args = (meanReturns, covMatrix, Days, riskFreeRate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = constraintSet
            bounds = tuple(bound for asset in range(numAssets))
            def negativeSR(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                    pReturns = np.sum(meanReturns*weights)*Days
                    pSTD = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                    return pReturns, pSTD
                pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Days)
                return - (pReturns - riskFreeRate)/pStd

            result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                                method='SLSQP', bounds=bounds, constraints=constraints)
            return result

        MaxSharpeRatio = maxSR(self.meanReturns, self.covMatrix, self.Days)

        def portfolioPerformance(weights, meanReturns, covMatrix, Days):
            WeightedReturns = np.sum(meanReturns*weights)*Days
            WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
            return WeightedReturns,  WeightedStd

        MSR_Returns, MSR_SD = portfolioPerformance(MaxSharpeRatio['x'], self.meanReturns, self.covMatrix, self.Days)
        MSR_allocation = pd.DataFrame(MaxSharpeRatio['x'], index=self.meanReturns.index, columns=['Allocation'])
        MSR_allocation.Allocation = [round(i*100,0) for i in MSR_allocation.Allocation]
            
        def minVar(meanReturns, covMatrix, Days, riskFreeRate = 0, constraintSet=(0,1)):
            "Minimize the negative SR, by altering the weights of the portfolio"
            numAssets = len(meanReturns)
            args = (meanReturns, covMatrix, Days, riskFreeRate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = constraintSet
            bounds = tuple(bound for asset in range(numAssets))
            def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                    WeightedReturns = np.sum(meanReturns*weights)*Days
                    WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                    return WeightedReturns,  WeightedStd
                return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
            result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            return result

        MinVol_Portfolio = minVar(self.meanReturns, self.covMatrix, self.Days)
        MinVol_Returns, MinVol_SD = portfolioPerformance(MinVol_Portfolio['x'], self.meanReturns, self.covMatrix, self.Days)
        MinVol_allocation = pd.DataFrame(MinVol_Portfolio['x'], index=self.meanReturns.index, columns=['Allocation'])
        MinVol_allocation.Allocation = [round(i*100,0) for i in MinVol_allocation.Allocation]
        #Efficient Frontier
        efficientList= []
        targetReturns= np.linspace(MinVol_Returns, MSR_Returns, 40)
        MSR_Returns , MinVol_Returns = round(MSR_Returns, 3), round(MinVol_Returns,3)
        MSR_SD , MinVol_SD = round(MSR_SD, 3), round(MinVol_SD,3)

        for target in targetReturns:
            def efficientOptimization(meanReturns, covMatrix, Days, returnTarget, constraintSet=(0,1)):
                numAssets = len(meanReturns)
                args= (meanReturns, covMatrix, Days)
                def portfolioReturns(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                    def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                        WeightedReturns = np.sum(meanReturns*weights)*Days
                        WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                        return WeightedReturns,  WeightedStd
                    return portfolioPerformance(weights, meanReturns, covMatrix, Days)[0]
                constraints= ({'type':'eq', 'fun': lambda x: portfolioReturns(x, meanReturns, covMatrix, Days)- returnTarget},
                                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bound = constraintSet
                bounds = tuple(bound for asset in range(numAssets))

                def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                    def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                        WeightedReturns = np.sum(meanReturns*weights)*Days
                        WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                        return round(WeightedReturns*100,2),  round(WeightedStd*100,2)
                    return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
                EffOpt= sc.minimize(portfolioVariance, numAssets*[1./numAssets],args=args, method= 'SLSQP', bounds=bounds, constraints=constraints)
                return EffOpt
            x = efficientOptimization(self.meanReturns, self.covMatrix, self.Days, target )
            efficientList.append(x['fun'])
        return MinVol_allocation, MinVol_Returns, MinVol_SD, MSR_allocation, MSR_Returns, MSR_SD, efficientList, targetReturns
#this works
    def EF_Graph(self, riskFreeRate=0, constraintSet=(0,1)):
        def calculatedResults(meanReturns, covMatrix, Days, riskFreeRate=0, constraintSet=(0,1)):
            """Read in Cov Matrix and other financial info and input max sharpe/ min volit/ and efficient frontier"""
            def maxSR(meanReturns, covMatrix, Days, riskFreeRate = 0, constraintSet=(0,1)):
                "Minimize the negative SR, by altering the weights of the portfolio"
                numAssets = len(meanReturns)
                args = (meanReturns, covMatrix, Days, riskFreeRate)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bound = constraintSet
                bounds = tuple(bound for asset in range(numAssets))
                def negativeSR(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                    def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                        pReturns = np.sum(meanReturns*weights)*Days
                        pSTD = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                        return pReturns, pSTD
                    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Days)
                    return - (pReturns - riskFreeRate)/pStd

                result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                                    method='SLSQP', bounds=bounds, constraints=constraints)
                return result
            
            MaxSharpeRatio = maxSR(self.meanReturns, self.covMatrix, self.Days)

            def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                WeightedReturns = np.sum(meanReturns*weights)*Days
                WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                return WeightedReturns,  WeightedStd

            MSR_Returns, MSR_SD = portfolioPerformance(MaxSharpeRatio['x'], self.meanReturns, self.covMatrix, self.Days)
            MSR_allocation = pd.DataFrame(MaxSharpeRatio['x'], index=self.meanReturns.index, columns=['Allocation'])
            MSR_allocation.Allocation = [round(i*100,0) for i in MSR_allocation.Allocation]
                
            def minVar(meanReturns, covMatrix, Days, riskFreeRate = 0, constraintSet=(0,1)):
                "Minimize the negative SR, by altering the weights of the portfolio"
                numAssets = len(meanReturns)
                args = (meanReturns, covMatrix, Days, riskFreeRate)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bound = constraintSet
                bounds = tuple(bound for asset in range(numAssets))
                def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                    def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                        WeightedReturns = np.sum(meanReturns*weights)*Days
                        WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                        return WeightedReturns,  WeightedStd
                    return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
                result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
                return result

            MinVol_Portfolio = minVar(self.meanReturns, self.covMatrix, self.Days)
            MinVol_Returns, MinVol_SD = portfolioPerformance(MinVol_Portfolio['x'], self.meanReturns, self.covMatrix, self.Days)
            MinVol_allocation = pd.DataFrame(MinVol_Portfolio['x'], index=self.meanReturns.index, columns=['Allocation'])
            MinVol_allocation.Allocation = [round(i*100,2) for i in MinVol_allocation.Allocation]
            efficientList= []

            targetReturns= np.linspace(MinVol_Returns, MSR_Returns, 40)
            MSR_Returns , MinVol_Returns = round(MSR_Returns, 3), round(MinVol_Returns,3)
            MSR_SD , MinVol_SD = round(MSR_SD, 3), round(MinVol_SD,3)

            for target in targetReturns:
                def efficientOptimization(meanReturns, covMatrix, Days, returnTarget, constraintSet=(0,1)):
                    numAssets = len(meanReturns)
                    args= (meanReturns, covMatrix, Days)
                    def portfolioReturns(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                        def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                            WeightedReturns = np.sum(meanReturns*weights)*Days
                            WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                            return WeightedReturns,  WeightedStd
                        return portfolioPerformance(weights, meanReturns, covMatrix, Days)[0]
                    constraints= ({'type':'eq', 'fun': lambda x: portfolioReturns(x, meanReturns, covMatrix, Days)- returnTarget},
                                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bound = constraintSet
                    bounds = tuple(bound for asset in range(numAssets))

                    def portfolioVariance(weights, meanReturns, covMatrix, Days, riskFreeRate = 0):
                        def portfolioPerformance(weights, meanReturns, covMatrix, Days):
                            WeightedReturns = np.sum(meanReturns*weights)*Days
                            WeightedStd = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(Days)
                            return WeightedReturns,  WeightedStd
                        return portfolioPerformance(weights, meanReturns, covMatrix, Days)[1]
                    EffOpt= sc.minimize(portfolioVariance, numAssets*[1./numAssets],args=args, method= 'SLSQP', bounds=bounds, constraints=constraints)
                    return EffOpt
                x = efficientOptimization(meanReturns, covMatrix, Days, target)
                efficientList.append(x['fun'])
            return MinVol_allocation, MinVol_Returns, MinVol_SD, MSR_allocation, MSR_Returns, MSR_SD, efficientList, targetReturns
        
        MinVol_allocation, MinVol_Returns, MinVol_SD, MSR_allocation, MSR_Returns, MSR_SD, efficientList, targetReturns = calculatedResults(self.meanReturns, self.covMatrix, self.Days, riskFreeRate=0, constraintSet=(0,1))
        
        MaxSharpeRatio= go.Scatter(
            name= 'Maximum Sharpe Ratio',
            mode='markers',
            x=[MSR_SD],
            y=[MSR_Returns],
            marker=dict(color='red', size=14,line=dict(width=3, color='black'))
        )
        MinVol= go.Scatter(
            name= 'Minimum Volitility',
            mode='markers',
            x=[MinVol_SD],
            y=[MinVol_Returns],
            marker=dict(color='green', size=14,line=dict(width=3, color='black'))
        )
        Efficient_Frontier= go.Scatter(
            name= 'Efficient Frontier',
            mode='lines',
            x=[round(ef_std,4) for ef_std in efficientList],
            y=[round(target,4) for target in targetReturns],
            line=dict(color='black',width=4, dash='dashdot')
        )
        data= [MaxSharpeRatio, MinVol, Efficient_Frontier]

        layout=go.Layout(
        title='Historical Efficient Frontier Optimization',
        yaxis=dict(title='Annualized Returns (%)'),
        xaxis=dict(title='Annualized Volitility (%)'),
            showlegend=True,
            legend=dict(
            x=0.75,
            y=0,
            traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
            width=800,
            height=600
            )
        fig = go.Figure(layout=layout, data=data)
        return fig.show()

    def MonteCarlo(self, Principal, OptimalWeighting=True, Simulations=100, Time=365, Seed=None, riskFreeRate=0):
        meanM=np.full(shape=(Time, len(self.weights)), fill_value=self.meanReturns)
        meanM=meanM.T
        portfolio_simulations= np.full(shape=(Time, Simulations), fill_value=0.0)
        initialInvestment= Principal


        def MC_Weights(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
            numAssets = len(meanReturns)
            args = (meanReturns, covMatrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = constraintSet
            bounds = tuple(bound for asset in range(numAssets))

            def Cholesky(weights, meanReturns, covMatrix):
                np.random.seed(seed=123)
                meanM=np.full(shape=(100, len(weights)), fill_value=meanReturns)
                meanM=meanM.T
                portfolio_simulations= np.full(shape=(100, 100), fill_value=0.0)
                for m in range(0, 100):
                    Z= np.random.normal(size=(100, len(weights)))
                    L= np.linalg.cholesky(covMatrix)
                    dailyReturns= meanM + np.inner( L, Z)
                    portfolio_simulations[:,m]= np.cumprod(np.inner(weights, dailyReturns.T)+1)
                    mean = portfolio_simulations.mean()
                return mean

            result = sc.minimize(Cholesky, numAssets*[1./numAssets], args=args,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            O_weights= result['x']
            return O_weights
        
        result = MC_Weights(self.meanReturns, self.covMatrix, constraintSet=(0,1))
        allocations= np.array(result)

        WeightedReturns_O = np.sum(self.meanReturns*allocations)*self.Days
        WeightedStd_O = np.sqrt(np.dot(allocations.T,np.dot(self.covMatrix, allocations)))*np.sqrt(self.Days)
        SR_O = round(((WeightedReturns_O- riskFreeRate)/ WeightedStd_O),2)

        WeightedReturns = np.sum(self.meanReturns*self.weights)*self.Days
        WeightedStd = np.sqrt(np.dot(allocations.T,np.dot(self.covMatrix, self.weights)))*np.sqrt(self.Days)
        SR = round(((WeightedReturns- riskFreeRate)/ WeightedStd),2)

        np.random.seed(seed=Seed)
        if OptimalWeighting == True:
            for m in range(0, Simulations):
                Z= np.random.normal(size=(Time, len(self.weights)))
                L= np.linalg.cholesky(self.covMatrix)
                dailyReturns= meanM + np.inner( L, Z)
                portfolio_simulations[:,m]= np.cumprod(np.inner(allocations, dailyReturns.T)+1)*initialInvestment
            mean = portfolio_simulations.mean()
            plt.plot(portfolio_simulations)
            plt.ylabel('Portfolio Value ($)')
            plt.xlabel("Days Ahead")
            #plt.text(1, initialInvestment*5, f'Sharpe Ratio:  {SR_O}', fontsize = 12, bbox = dict(facecolor = 'grey', alpha = 0.5))
            plt.title('MC Simulations of Current Weighted Portfolio\n Average Final Value:  '+ str(round(mean,0) ))
            plt.show()
        else:
            for m in range(0, Simulations):
                Z= np.random.normal(size=(Time, len(self.weights)))
                L= np.linalg.cholesky(self.covMatrix)
                dailyReturns= meanM + np.inner( L, Z)
                portfolio_simulations[:,m]= np.cumprod(np.inner( self.weights, dailyReturns.T)+1)*initialInvestment
            mean = portfolio_simulations.mean()
            plt.plot(portfolio_simulations)
            plt.ylabel('Portfolio Value ($)')
            plt.xlabel("Days Ahead")
            plt.text(1, initialInvestment*5, f'Sharpe Ratio:  {SR}', fontsize = 12, bbox = dict(facecolor = 'grey', alpha = 0.5))
            plt.title('MC Simulations of an Optimal Portfolio\n Average Final Value:  '+ str(round(mean,0)))
            plt.show()
        
        



length=365
start_date= dt.datetime(2020,1, 1)
end_date = start_date+dt.timedelta(days=length)
Stock_names = ['AMZN', 'MSFT', "GOOG"]
weights= np.random.random(len(Stock_names))
weights /= np.sum(weights)
portfolio1= Portfolio(Stock_names, start_date, end_date, weights)
#3 is the allocation
#print(portfolio1.calculatedResults(.5)[7])
df= pd.DataFrame(portfolio1.calculatedResults(.5)[7],portfolio1.calculatedResults(.5)[6])
#print(portfolio1.EF_Graph(.5))
#print(portfolio1.MonteCarlo(10000, OptimalWeighting=False))
#print(portfolio1.MonteCarlo(10000, OptimalWeighting=True))
portfolio1.MonteCarlo(10000)