# IMPORTING PLOTTIN LIBRARIES
import matplotlib.pyplot as plt
plt.style.reload_library()
plt.style.use(['science', 'grid'])

# IMPORTING SUPPORT PACKAGES
import pandas as pd
import numpy as np
from ipywidgets import interact, widgets

##########################################################################################################
# FUNCTION THAT PLOTS A HISTOGRAM WITH OPTIMAL BIN SIZE 
# MORE INFO ABOUT HOW TO CALCULATE THE OPTIMAL BIN SIZE: https://www.neuralengine.org//res/histogram.html 
##########################################################################################################
def plotHistogram(x, title):
    
    # Making Histogram Legend
    ENTRIES = len(x)
    MEAN = x.mean()
    STD = x.std()
    label = r'$\textbf{Entries:}$ ' + str("{:}".format(ENTRIES)) + '\n' + r'$\textbf{Mean:}$ ' + str("{:.4}".format(MEAN)) + '\n' +  r'$\textbf{Std Dev}$: ' + str("{:.3}".format(STD))
    
    # Useful arrays for calculating optimal number of bins
    x_max, x_min, N_MIN, N_MAX = x.max(), x.min(), 2, 100
    N = np.array(range(N_MIN,N_MAX)) # of Bins
    D = (x_max-x_min)/N    #Bin size vector
    C = np.zeros(shape=(np.size(D),1))
    
    # Computation of the cost function
    for i in range(np.size(N)):
        edges = np.linspace(x_min,x_max,N[i]+1) # Bin edges
        ki = np.histogram(x,edges) # Count # of events in bins
        ki = ki[0]
        k = np.mean(ki) #Mean of event count
        v = sum((ki-k)**2)/N[i] #Variance of event count
        C[i] = (2*k-v)/((D[i])**2) #The cost Function
        
    # Optimal Bin Size Selection
    cmin = min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0])
    optD = D[idx]
    
    # Plotting
    edges = np.linspace(x_min,x_max,N[idx]+1)
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.hist(x,edges,facecolor = 'g', edgecolor= "black", linewidth=1 , label=label)
    plt.xticks(fontsize=25)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.yticks(fontsize=25)
    title = title.replace('_', '\_')
    plt.ylabel(r'$\textit{'+ 'Entries' + '}$', fontsize=30) 
    plt.title(title, fontsize=30)
    plt.legend(prop={'size': 16}, loc=1) 
    plt.show()
##########################################################################################################



##########################################################################################################
# FUNCTION THAT PLOTS EACH ELEMENT IN OUR BRANCH AS TABS
##########################################################################################################
def TABS_PLOTTER(df, branch):
    
    # Generating Output type variables according to the number N of items in our branch
    tab_contents = [str(i) for i in range(df.shape[1])]
    var_names = ['out'+name for name in tab_contents]
    new_var = []
    for var in var_names:
        globals()[var] = widgets.Output(layout={})
        new_var.append(globals()[var])

    # Generating tab object with N children, each containing an empty display box
    tab = widgets.Tab(children=new_var)

    # Tabs titles
    for i in range(len(tab_contents)):
        tab.set_title(i, str(i))

    # Filling each tab with a plot
    for i, j in zip(new_var, tab_contents):
        with i:
            plotHistogram(df[int(j)], branch + '[' + j + ']')

    # Displaying dynamic tabs
    tab.observe(tab_toggle_var)
    #tab_toggle_var()
    display(tab)

def tab_toggle_var(*args):
    global vartest
    if tab.selected_index ==0:
        vartest = 0
    else:
        vartest = 1
##########################################################################################################