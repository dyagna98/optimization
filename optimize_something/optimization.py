"""MC1-P2: Optimize a portfolio.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Divya Yagnamurthy (replace with your name)
GT User ID: dyagnamurthy3 (replace with your User ID)
GT ID: 903195460 (replace with your GT ID)
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import scipy.optimize as spo

# This is the function that will be tested by the autograder  		   	  			  	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		   	  			  	 		  		  		    	 		 		   		 		  
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=True):
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		   	  			  	 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)
    prices = prices_all[syms]  # only portfolio symbols
    norm_prices= prices/prices.iloc[0]
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    norm_SPY = prices_SPY / prices_SPY.iloc[0]

    alloc_guesses = np.asarray([1.0/len(syms)] * len(syms))
    # find the allocations for the optimal portfolio  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case
    bounds = [(0.0, 1.0)] * len(syms)
    constraint = {'type': 'eq', 'fun': lambda inputs: 1.0- np.sum(inputs)}
    optimizer = spo.minimize(sharpe_func, alloc_guesses, args=(norm_prices, ), method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraint)
    allocs = optimizer.x # add code here to find the allocations
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    allocated = norm_prices * allocs
    port_val = allocated.sum(axis=1)
    dailyReturn = dailyReturns(port_val)
    cr = (port_val[-1]/port_val[0]) - 1
    adr = dailyReturn.mean()
    sddr = dailyReturn.std()
    sr = np.math.sqrt(252) * (adr / sddr)
    # Get daily portfolio value  		   	  			  	 		  		  		    	 		 		   		 		  
    # port_val = prices_SPY # add code here to compute daily portfolio values
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		   	  			  	 		  		  		    	 		 		   		 		  
    if gen_plot:
        # add code to plot here  		   	  			  	 		  		  		    	 		 		   		 		  
        df_temp = pd.concat([port_val, norm_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        plt.grid()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Daily Portfolio Value & SPY")
        plt.savefig("optimization.png")
        plt.show()

    return allocs, cr, adr, sddr, sr

def sharpe_func(alloc_guesses, norm_prices):
    allocated = norm_prices * alloc_guesses
    portVal = allocated.sum(axis=1)

    daily_returns = dailyReturns(portVal)

    sr = np.math.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    return(sr*-1)

def dailyReturns(portVal):
    dR = portVal.copy()
    dR[1:] = (portVal[1:]/portVal[:-1].values) -1
    dR.iloc[0] = 0
    dR = dR[1:]
    return dR
  		   	  			  	 		  		  		    	 		 		   		 		  
def test_code():  		   	  			  	 		  		  		    	 		 		   		 		  
    # This function WILL NOT be called by the auto grader  		   	  			  	 		  		  		    	 		 		   		 		  
    # Do not assume that any variables defined here are available to your function/code  		   	  			  	 		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		   	  			  	 		  		  		    	 		 		   		 		  
    # the autograder!  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2009,1,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2010,1,1)  		   	  			  	 		  		  		    	 		 		   		 		  
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		   	  			  	 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Print statistics  		   	  			  	 		  		  		    	 		 		   		 		  
    # print(f"Start Date: {start_date}")
    # print(f"End Date: {end_date}")
    # print(f"Symbols: {symbols}")
    # print(f"Allocations:{allocations}")
    # print(f"Sharpe Ratio: {sr}")
    # print(f"Volatility (stdev of daily returns): {sddr}")
    # print(f"Average Daily Return: {adr}")
    # print(f"Cumulative Return: {cr}")
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		   	  			  	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		   	  			  	 		  		  		    	 		 		   		 		  
    test_code()  		   	  			  	 		  		  		    	 		 		   		 		  
