# Many amateur traders tend to focus on total returns rather than risk-adjusted returns. 
# Total returns by itself are quite meaningless and do not accurately provide a true 
# measure of a system’s performance. Risk-adjusted returns, on the other hand, 
# is a much more valuable way to assess system performance.

# https://tradingsim.com/blog/how-to-measure-your-trading-performance/#2_-_Maximum_Draw_Down
# https://www.ea-coder.com/5-trading-metrics-explained/

# Studies have shown that 85% - 90% of day traders fail at turning a consistent profit.
 
#Stop obsessing over the dozens of trading performance indicators and reports.  
#You need to establish very basic trading performance metrics centered around profitability and measure these in short sprints.

# portfolio: portfolio based on current strategy: numpy array
# close: close-price of asset: numpy array
# invested: moment of investement [True or False]
#%% Libraries
import numpy as np
from caerus.utils.ones2idx import ones2region

#%%
def risk_performance_metrics(perf):
    # Setup values for computations
    if not np.any(perf.columns=='portfolio_value'):
        perf['portfolio_value']=None
#    if not np.any(perf.columns=='max_drawdown'):
#        perf['max_drawdown']=None
        
    if np.any(perf.columns=='positions'):
        #zipline
        positions = np.array(list(map(lambda x: x!=[], perf.positions.values)), dtype='int')
    elif np.any(perf.columns=='invested'):
        #our approach
        positions = perf.invested
    else:
        print('Positions can not be retrieved.')
        return None

    portfolio_value = perf.portfolio_value.values
    close = perf.asset.values

    # Determine trading boundaries
#    trade_boundaries=np.diff(positions,1)  
#    trades_start=np.where(trade_boundaries==1)[0]
#    trades_stop=np.where(trade_boundaries==-1)[0]
#    idx_trades=list(zip(trades_start,trades_stop))
    idx_trades=ones2region(positions)
    
    # Retrieve amount win/lose for the particular trade
    trades_values = np.array(list(map(lambda x: portfolio_value[x[1]]-portfolio_value[x[0]], idx_trades)))
    trades_values = trades_values[np.isnan(trades_values)==False]
    trades_win=[]
    trades_lose=[]

    if np.any(trades_values>0):
        trades_win = trades_values[trades_values>0]
    if np.any(trades_values<=0):
        trades_lose = trades_values[trades_values<=0]
        

    # Compute various performances
    out = dict()
    out['ratio_profit']         = ratio_profit(portfolio_value)
    out['ratio_buy_and_hodl']   = ratio_profit(close)
    out['expectancy']           = expectancy(trades_win,trades_lose)
    out['largest_losing_trade'] = largest_losing_trade(trades_lose)
    out['win_percentage']       = win_percentage(trades_win,trades_lose)
    out['profit_factor']        = profit_factor(trades_win,trades_lose)
    out['maximum_drawdown']     = maximum_drawdown(perf, idx_trades)
    out['number_of_trades']     = len(idx_trades)
    out['winning_trades']       = len(trades_win)
    out['losing_trades']        = len(trades_lose)
    out['winning_balance']      = np.sum(trades_win)
    out['losing_balance']       = np.sum(trades_lose)
    out['Total_balance']        = np.sum(trades_values)
    
    return(out)

#%% Metrics
def ratio_profit(portfolio_value):
    #perf.algorithm_period_return
    portfolio_value=portfolio_value[~np.isnan(portfolio_value)]
    return (portfolio_value[-1]/portfolio_value[0])-1

def win_percentage(trades_win,trades_lose):
    # Win Percentage tells us how many winning trades that we have in relation to the total number of trades taken.
    # For example: 77 Winning Trades / 140 Total Trades = 55% Win Percentage
    out = 0
    if (len(trades_win)+len(trades_lose))>0:
        out = len(trades_win) / (len(trades_win)+len(trades_lose))
    return(out)

def expectancy(trades_win,trades_lose):
    # Evaluating a trading strategy or system at any given account size
    # Expectancy = (Winning Percentage x Average Win Size) – (Losing Percentage x Average Loss Size)
    expectcy       = 0
    winningPerc    = 0
    losingPerc     = 0
    trades_win_nr  = len(trades_win)
    trades_lose_nr = len(trades_lose)
    
    if (trades_lose_nr+trades_win_nr)>0:
        winningPerc = trades_win_nr / (trades_lose_nr+trades_win_nr)
    if (trades_lose_nr+trades_win_nr)>0:
        losingPerc = trades_lose_nr / (trades_lose_nr+trades_win_nr)
    
    if len(trades_win)>0 and len(trades_lose)>0:
        expectcy = (winningPerc*np.nan_to_num(np.mean(trades_win))) - (losingPerc*np.nan_to_num(np.mean(trades_lose)))

    return(expectcy)

def largest_losing_trade(trades_lose):
    # It will help in system design and stress testing of your strategy. Your largest losing trade is self-explanatory and measures the amount of the single largest loss incurred.
    # We want to keep our largest loss within a reasonable range, so as not to jeopardy blowing up our account. It is essential to know the reason behind your largest loss so that you can try to contain it in the future.
    out=0
    if len(trades_lose)>0:
        out=np.min(trades_lose)
    return(out)

def profit_factor(trades_win,trades_lose):
    # R is the profit factor which takes the total of your winning trades divided by the absolute value of your losing trades to determine your profitability.
    # So, if I have a winning percentage of only 40%, buy my winners are much larger than my losers, I will still turn a profit.  For example, take a look at the below table to further illustrate this point:
    #
    #10 Total Trades	
    #4 sum Winners	19900
    #6 sum Losers	12034
    #R=	19900/12034 = 1.65
    #
    # For example, let’s say that during the course of a year, your winning trades resulted in $ 26,500 profits and your losing trades resulted in $15,250 in losses.
    # Then based on this, your profit factor would be:
    # $26,500 / $ 15,250 = 1.74
    #
    # profit_factor
    # [<1] trading system or strategy is unprofitable, 
    # [1.10-1.40] is considered a moderately profitable system, 
    # [1.41-2.00] is quite good, and finally, 
    # [>2.01] is considered excellent.
    #

    losers_sum=0
    winners_sum=0
    out=0

    if len(trades_win)>0:
        winners_sum=np.sum(trades_win)
    if len(trades_lose)>0:
        losers_sum=np.abs(np.sum(trades_lose))
    if losers_sum>0:
        out=winners_sum/losers_sum
    else:
        out=2.99 # magic number
    
    if (winners_sum==0) and (losers_sum==0):
        out=0
        
    return(out)
    
def maximum_drawdown(perf, idx_trades):
    # Evaluate risk-adjusted returns
    # Maximum draw down represents how much money you lose from a recent account high before making a new high in your account.  For example, if you just had a high of $100,000 and then pull back to $85,000 before exceeding $100,000, then you would have a maximum draw down of 15% ($85,000/%100,000).  The maximum draw down is probably the most valuable key performance indicator you should monitor when it comes to trading performance.  
    # Minimize your draw downs may be one of the most important factors!
    
    # Maximum Drawdown measures a portfolio’s largest peak to valley decline prior to a new peak in the equity curve
    # Assume your trading account started at $10,000 and increased to $15,000 over a certain period, and then fell to $7,000 after a string of losing trades. 
    # Later, it rebounded a bit increasing to $9,000. Soon after, another string of losses resulted in the account falling to $ $6,000. 
    # After some time, you were able to get back on track and get the account level to $ 18,000.

    # The answer is: 40% was your max drawdown.
    # $6,000 (lowest low prior to new peak ) / $15,000 ( highest peak prior to a new peak )
    maxdrawdown=None
    if np.any(perf.columns=='returns'):
        maxdrawdown=np.zeros(len(idx_trades))
        for i in range(0, len(idx_trades)):
            high=np.nanmax(perf.returns.iloc[idx_trades[i][0]:idx_trades[i][1]+2])
            low=np.nanmin(perf.returns.iloc[idx_trades[i][0]:idx_trades[i][1]+2])
            if high!=0:
                maxdrawdown[i]=low/high
        
        maxdrawdown = maxdrawdown[~np.isin(maxdrawdown, [-np.inf, np.inf, np.nan])]
        maxdrawdown = np.nanmean(maxdrawdown)

    return(maxdrawdown)


