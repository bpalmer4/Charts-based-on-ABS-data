# finalise_plot.py

# --- imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --- constants
DEFAULT_FILE_TYPE = 'png'
DEFAULT_FIG_SIZE = (9, 4.5)


# --- utility functions

def _apply_kwargs(ax, **kwargs):
    
    fig = ax.figure
    
    rfooter = 'rfooter'
    if rfooter in kwargs:
        fig.text(0.99, 0.001, 
            kwargs[rfooter],
            ha='right', va='bottom',
            fontsize=9, fontstyle='italic',
            color='#999999')
    
    lfooter = 'lfooter'
    if lfooter in kwargs:
        fig.text(0.01, 0.001, 
            kwargs[lfooter],
            ha='left', va='bottom',
            fontsize=9, fontstyle='italic',
            color='#999999')

    figsize = 'figsize'  
    if figsize in kwargs:
        print(kwargs['figsize'])
        fig.set_size_inches(*kwargs[figsize])
    else:
        fig.set_size_inches(*DEFAULT_FIG_SIZE) 
        
    xlabel = 'xlabel'
    if xlabel in kwargs:
        ax.set_xlabel(kwargs[xlabel])
        
    concise_dates = 'concise_dates'
    if concise_dates in kwargs:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=0, horizontalalignment='center')
            

### --- main function
        
def finalise_plot(ax, title, ylabel, tag, chart_dir, **kwargs): 
    """Function to finalise plots
        Arguments:
        - ax - matplotlib axes object
        - title - string - plot title, also used to save the file
        - ylabel - string - 
        - tag - string - used in file name to make similar plots have unique file names
        - chart_dir - string - location of the chartr directory 
        kwargs
        - file_type - string - specify a file type - eg. 'png' or 'svg'
        - lfooter - string - text to display on bottom left of plot
        - rfooter - string - text to display of bottom right of plot
        - figsize - tuple - figure size in inches - eg. (8, 4)
        - show - Booolean - whether to show the plot or not
        - concise_dates - use the matplotlib concise dates formatter
        Returns: 
        - None
    """
    
    # annotate plot
    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    
    # fix margins - I should not need to do this!
    #FACTOR = 0.015
    #xlim = ax.get_xlim()
    #adjustment = (xlim[1] - xlim[0]) * FACTOR
    #ax.set_xlim([xlim[0] - adjustment, xlim[1] + adjustment])
    
    # apply keyword arguments
    _apply_kwargs(ax, **kwargs)
    
    # finalise
    fig = ax.figure
    fig.tight_layout(pad=1.1)
    
    # save and close
    title = title.replace(":", "-")
    file_type = DEFAULT_FILE_TYPE
    if 'file_type' in kwargs:
        file_type = kwargs['file_type']
    fig.savefig(f'{chart_dir}/{title}-{tag}.{file_type}', dpi=125)
    if 'show' in kwargs and kwargs['show']:
        plt.show()
    plt.close()
    
    return None
