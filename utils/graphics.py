import os

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, NullFormatter

def line_plot(x, ys, path=None, line_legends=None,
              xlabel=None, ylabel=None, title=None, xlog=False, ylog=False,
              linestyles = [':', '--', '-.', (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10))],
              colors = ['#000000', '#FF0000', '#360CE8', '#4ECE00', '#E8A10C', '#808080', 'pink'],
              markers = ['o','x','+','d','1'],
              figsize=(9, 5),
              ylim=None,
              grid=True,
              yticks=None,
              xticks=None):

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    if markers is None:
        markers = ["None"]*len(ys)

    if linestyles is None:
        linestyles = ['-']*len(ys)
    
    if line_legends is None:
        line_legends = [None]*len(ys)
    
    lines = []
    for i,y in enumerate(ys):            
        lines.append(            
            ax.plot(x, y, 
                linestyle=linestyles[i],
                linewidth=1.5, 
                color=colors[i],
                label=line_legends[i],
                    marker=markers[i]
            )
        )

    if line_legends[0] is not None:
        plt.legend()
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid is True:
        ax.grid(True)
    if yticks is not None:
        ax.set_yticks(yticks)
    if ylog is True:
        plt.yscale('log')    
    if xlog is True:
        plt.xscale('log')    
    if xticks is not None:
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        # ax.xaxis.set_major_formatter(NullFormatter())
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])

    plt.show()

    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        fig.savefig(path, dpi=300)
        print('Graphic saved at: ' + path)
    # else:
    plt.clf()
    plt.close()


def epsilon_bar_plot(series, std, es, random, title, path=None, legend_path=None):
    ind = np.arange(5) 
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 7), tight_layout=True)
    
    # fig, ax = plt.subplots(tight_layout=True)
    rects1 = ax.bar(ind - (3*width)/2, tuple(series[0]), width, #yerr=std[0],
                    color='#4ECE00', label='$TopInf_{lap}$ $\deg_{max} = n-1$')
    rects2 = ax.bar(ind - (1*width)/2, series[1], width, #yerr=std[1],
                    color='#E8A10C', label='$TopInf_{lap}$ $\deg_{max} = \Delta(G)$')
    rects3 = ax.bar(ind + (1*width)/2, series[2], width, #yerr=std[2],
                    color='#FF0000', label='$TopInf_{exp}$ $\deg_{max} = n-1$')
    rects4 = ax.bar(ind + (3*width)/2, series[3], width, #yerr=std[3],
                    color='#360CE8', label='$TopInf_{exp}$ $\deg_{max} = \Delta(G)$')

    if legend_path:
        figlegend = plt.figure(figsize=(8.9, 0.25), tight_layout=True)
        figlegend.legend((rects1, rects2, rects3, rects4), 
            ('Laplace $\Delta f = n-1$', 
            'Laplace $\Delta f = \Delta(G)$', 
            'Exponential $\Delta u = n-1$', 
            'Exponential $\Delta u = \Delta(G)$'), 'center', ncol=4)
        figlegend.savefig(legend_path, dpi=500)
        figlegend.clf()
        plt.close(figlegend)
    # figlegend.close()


    ax.plot([0-(5*width)/2,4+(5*width)/2],[random, random], linestyle=(0, (5, 5)), linewidth=1., color='black')

    ax.set_ylabel('Recall')
    # ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(['$\epsilon = %s$' % (e) for e in es])
    # ax.legend(loc=2, fontsize='x-small')
    ax.grid(True)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    plt.xlim(0-(5*width)/2,4+(5*width)/2)
    if path:
        fig.savefig(path, dpi=300)
    else:
        fig.show()
    plt.clf()
    plt.close()
    # exit(0)

def bar_plot(xs, bar_legend=None, xlabel=None, ylabel=None, xtick_labels=None, 
            legend_path=None, path=None, title=None, random=None, labels=None,
            colors = ['#360CE8', '#FF0000', '#4ECE00', '#E8A10C']):
    bar_n = len(xs)
    group_n = len(xs[0])
    ind = np.arange(group_n)

    total_width = 1.
    width = total_width/bar_n
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    plt.yscale('log')    
    bars = []
    if labels is None:
        for i,x in enumerate(xs):
            bars.append(ax.bar(ind + (width/2) + width*i, x, width,
                    color=colors[i]))    
    else:   
        for i,x in enumerate(xs):
            bars.append(ax.bar(ind + (width/2) + width*i, x, width,
                    color=colors[i], label=labels[i]))    

    if legend_path:
        figlegend = plt.figure(figsize=(8.9, 0.25), tight_layout=True)
        figlegend.legend(bars, bar_legend, 'center', ncol=len(bars))
        figlegend.savefig(legend_path)
        figlegend.clf()
        plt.close(figlegend)
    elif labels is not None:
        plt.legend()

    if random:
        ax.plot([0-(1-total_width),group_n],[random, random], linestyle=(0, (5, 5)), linewidth=1., color='black')
        plt.xlim(0-(1-total_width),group_n)

    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if xtick_labels:
        ax.set_xticks(ind+total_width/2)
        ax.set_xticklabels(xtick_labels)
    ax.grid(True)
    if path:
        fig.savefig(path)
    else:
        plt.show()
    plt.clf()
    plt.close()
    
    
def line_plot1(xys, path=None, line_legends=None, legend_path=None, 
              xlabel=None, ylabel=None, title=None, xlog=False, ylog=False,
              linestyles = [':', '--', '-.', (0, (3, 10, 1, 10)), (0, (3, 10, 1, 10, 1, 10))],
              colors = ['#000000', '#FF0000', '#360CE8', '#4ECE00', '#E8A10C', '#808080', 'pink'],
              markers = ['o','x','+','d','1'],
              figsize=(9, 5),
              ylim=None):

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    if markers is None:
        markers = ["None"]*len(xys)

    if linestyles is None:
        linestyles = ['-']*len(xys)
    
    if line_legends is None:
        line_legends = [None]*len(xys)
    
    lines = []
    for i,(x,y) in enumerate(xys):    
        lines.append(
            ax.plot(x, y, 
                linestyle=linestyles[i],
                linewidth=1.5, 
                color=colors[i],
                label=line_legends[i],
                marker=markers[i]
            )
        )

    if line_legends[0] is not None:
        plt.legend()
    if ylog is True:
        plt.yscale('log')    
    if xlog is True:
        plt.xscale('log')    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=500)
        print('Graphic saved at: ' + path)
    else:
        plt.show()
    plt.clf()
    plt.close()


def histogram(x, title="", xlabel="", ylabel="Frequency", path=None, 
                log=True, range_x=None, fig_size=(9,4)):
    # plt.figaspect([10, 4])
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    plt.hist(x, color="blue", bins=50, range=range_x)
    plt.xlim(left=0.)
    # plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)    
    plt.grid(True)
    if log == True:
        plt.yscale('log')    
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=500)
        print('Histogram saved at: ' + path)
    else:
        plt.show()
    plt.clf()
    plt.close()

def histogram1(x1, x2, range_x=None, path=None, title1=None, title2=None):
    fig, axis = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), tight_layout=True)
    axis[0].hist(x1, color="red", bins=50, range=range_x)
    if title1:
        axis[0].set_title(title1)
    if title2:
        axis[1].set_title(title2)

    axis[1].hist(x2, color="blue", bins=50, range=range_x)

    if path:
        plt.savefig(path, dpi=500)
    else:
        plt.show()

def box_plot(xs, labels=None, path=None,
            xlabel=None, ylabel=None, sym='.'):
    
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)

    ax.boxplot(xs, 
            sym=sym)

    if labels is not None:
        ax.set_xticklabels(labels,
            rotation=90, fontsize=8)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    ax.yaxis.grid(True)
    if path:
        plt.savefig(path, dpi=500)
    else:
        plt.show()
    plt.clf()
    plt.close()

def mesh(M, cmap="jet", xticks=None, yticks=None, plot_values=False, path=None,
        ylabel=None, xlabel=None, title=None):
    fig, ax = plt.subplots(tight_layout=True, sharex=True, sharey=False)
    # ax.set_adjustable('box-forced')
    im = ax.imshow(M, cmap=cmap, aspect='auto')
    # ax.autoscale(False)
    # ax.set_axis_off()

    
    fig.colorbar(im, ax=ax, shrink=0.55)

    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)

    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)
    font = {
        'size': 20,
        'color':  'black',
    }
    if plot_values:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                text = ax.text(j, i, M[i, j],
                                ha="center", va="center", color="w", fontdict=font)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()
    plt.close()


def scatter_plot(x, y, path=None, figsize=(9, 5),
                xlabel=None, ylabel=None, 
                title=None, xlog=False, ylog=False,
                ylim=None, c=None, s=None, cmap=None):
    
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    ax.scatter(x, y, c=c, s=s, cmap=cmap)

    if ylog:
        plt.yscale('log')    
    if xlog:
        plt.xscale('log')    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True)
    
    if path:
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(path, dpi=500)
        print('Graphic saved at: ' + path)
    else:        
        plt.show()
    plt.clf()
    plt.close()
    
if __name__ == "__main__":
    # bar_plot([[1,2,4],[4,5,6]], 
    #          bar_legend=['a','b'],
    #          legend_path='teste.pdf',
    #          path='teste1.pdf',
    #          xlabel='xlabel',
    #          ylabel='ylabel',
    #          xtick_labels=['1','2','3'],
    #          title='title')
    # line_plot([1,2,3],[[1,2,3],[4,5,6]],
    #         line_legends=['a','b'],
    #         legend_path='legend.pdf',
    #         path='image.pdf',
    #         xlabel='xlabel',
    #         ylabel='ylabel',
    #         title='title')
    # box_plot([[1,2,3,4],[4,5,6,7],[2,3,5,3],[5,2,6,8],[5,3,5,4,9]],
            # labels=["x1","x2","x3","x4","x5"])
    mesh(np.array([[1,2],[3,4]]), 
                    xticks=['a','b'], 
                    yticks=['c','d'],
                    plot_values=True)
