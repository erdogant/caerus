"""This code is part of caerus and is not designed for usage in other code."""
#--------------------------------------------------------------------------
# Name        : caerus.py
# Author      : E.Taskesen
# Contact     : erdogant@gmail.com
# Date        : May. 2020
#--------------------------------------------------------------------------

import caerus.utils.csutils as csutils
import matplotlib.pyplot as plt
import numpy as np

# %%
def _plot_graph(out, figsize=(15,8)):
    df = out['df']
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['X'].loc[df['peak']], 'or', linewidth=1)
    ax.plot(df['X'].loc[df['valley']], 'og', linewidth=1)
    ax.plot(df['X'], 'k', linewidth=0.5)
    ax.vlines(np.where(df['peak'])[0], df['X'].min(), df['X'].max(), linestyles='--', colors='r')
    ax.vlines(np.where(df['valley'])[0], df['X'].min(), df['X'].max(), linestyles='--', colors='g')
    
    uilabx = np.unique(df['labx'])
    for labx in uilabx:
        if labx>0:
            Iloc = df['labx']==labx
            plt.plot(np.where(Iloc)[0], df['X'].loc[Iloc].values, marker='.', linewidth=0.1)

    ax.set_ylabel('Input value')
    ax.set_xlabel('Time')
    ax.grid(True)

    return ax

# %%
def _plot_gridsearch(out, figsize=(15,8)):
    # Make figure
    [fig,(ax1,ax2)]=plt.subplots(2,1, figsize=figsize)

    # Make plots across the gridsearch
    for k in range(0,len(out['window'])):
        for i in range(0,len(out['minperc'])):
            # label = list(map(( lambda x: 'window_' + x), windows.astype(str)))
            ax1.plot(out['minperc'], out['balances'][:,k], label='window_' + str(out['window'][k]))
            ax2.plot(out['minperc'], out['trades'][:,k], label='window_' + str(out['window'][k]))

    # ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel('Percentage')
    ax1.set_ylabel('winning_balance')
    ax2.grid(True)
    ax2.set_xlabel('Percentage')
    ax2.set_ylabel('Nr Trades')
    plt.show()

# Make plot
def _plot(out, threshold=0.25, figsize=(25,15)):
    df = out['X']
    loc_start = out['loc_start']
    loc_stop = out['loc_stop']
    loc_start_best = out['loc_start_best']
    loc_stop_best = out['loc_stop_best']
    simmat = out['simmat']

    # Top plot
    # agg = out['agg']
    fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=figsize)
    # Make heatmap
    ax1.matshow(np.flipud(simmat.T))
    ax1.set_aspect('auto')
    # ax1.gca().set_aspect('auto')
    ax1.grid(False)
    ax1.set_ylabel('Perc.difference in window\n(higher=better)')
    ax1.set_xlim(0,simmat.shape[0])

    xlabels = simmat.columns.values.astype(str)
    Iloc=np.mod(simmat.columns.values,10)==0
    xlabels[Iloc==False]=''
    xlabels[-1]=simmat.columns.values[-1].astype(str)
    xlabels[0]=simmat.columns.values[0].astype(str)
    ax1.set_yticks(range(0,len(xlabels)))
    ax1.set_yticklabels(np.flipud(xlabels))
    ax1.grid(True, axis='x')

    # make aggregated figure
    # Normalized successes across the n windows for percentages above minperc.
    # 1 depicts that for location i, all of the 1000 windows of different length was succesfull in computing a percentage above minperc
    [outagg, I] = csutils._agg_scores(simmat, threshold)
    ax2.plot(outagg)
    ax2.grid(True)
    ax2.set_ylabel('Cummulative\n#success windows')
    ax2.set_xlim(0,simmat.shape[0])
    ax2.hlines(threshold,0,simmat.shape[0], linestyles='--', colors='r')

    # Plot local maxima
    if loc_stop_best is not None:
        ax2.vlines(loc_stop_best,0,1, linestyles='--', colors='r')
        ax3.plot(df.iloc[loc_stop_best],'or', linewidth=1)
    # Plot local minima
    if loc_start_best is not None:
        ax2.vlines(loc_start_best,0,1, linestyles='--', colors='g')
        ax3.plot(df.iloc[loc_start_best],'og', linewidth=1)

        for i in range(0,len(loc_start)):
            ax3.plot(df.iloc[np.arange(loc_start[i][0],loc_start[i][1])],'g', linewidth=2)
            for k in range(0,len(loc_stop[i])):
                ax3.plot(df.iloc[np.arange(loc_stop[i][k][0],loc_stop[i][k][1])],'r', linewidth=2)
    # Plot region-minima-maxima
    ax3.plot(df,'k', linewidth=1)

    ax3.set_ylabel('Input value')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    ax3.set_xlim(0,simmat.shape[0])
    if loc_start_best is not None:
        ax3.vlines(loc_start_best,df.min(),df.max(), linestyles='--', colors='g')
    if loc_stop_best is not None:
        ax3.vlines(loc_stop_best,df.min(),df.max(), linestyles='--', colors='r')
    plt.show()
    return fig
