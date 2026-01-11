import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def normalized_Y(Y):
  col_sum = Y.sum(axis=0)
  Y = Y / col_sum[np.newaxis, :]
  return Y

def plot_multi_stack_bar(plot_params, my_params, Y1, Y2, labels, xlabel, ylabel, anchor=None, figpath=None):
  # print(plt.rcParams.keys())
    pylab.rcParams.update(plot_params)  #更新自己的设置
    plt.rcParams['pdf.fonttype'] = 42

    width = my_params['bar_width']
    colors = my_params['colors']
    hatchs = my_params['hatchs']

    fig, ax = plt.subplots()
    assert Y1.shape[1] == Y2.shape[1]
    n = Y1.shape[1]
    gap = 0

    ############## stack bar 1
    Y = Y1
    ind = np.arange(n) - width/2 - gap/2              # the x locations for the groups
    pre_bottom = np.zeros(len(Y[0]))
    h_legends = []
    e_legends = []
    for i, y in enumerate(Y):
        leg1 = plt.bar(ind,y,width,color=colors[i],  hatch=hatchs[i], bottom=pre_bottom, linewidth=params['lines.linewidth'], edgecolor='white')
        leg2 = plt.bar(ind, y, width, color='none', bottom=pre_bottom, lw=0.7, edgecolor='black')
        h_legends.append(leg1)
        e_legends.append(leg2)
        pre_bottom += y  

    fontsize = 6
    text_offset = 0.1
    for x,y in zip(ind, Y1[0]):
        plt.text(x-text_offset, 101, "mode1",horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
        ax.annotate(f'{y/100:.1%}', xy=(x,-0.55), xytext=(x-.5,-16),arrowprops=dict(arrowstyle="->",color='C3'), color='C3',fontsize=fontsize)


    ############## stack bar 2
    Y = Y2
    ind = np.arange(n) + width/2 + gap/2             # the x locations for the groups
    pre_bottom = np.zeros(len(Y[0]))
    flag = True
    for i, y in enumerate(Y):
        plt.bar(ind,y,width,color=colors[i], hatch=hatchs[i], bottom=pre_bottom, linewidth=params['lines.linewidth'], edgecolor='white')
        plt.bar(ind, y, width, color='none', bottom=pre_bottom, lw=.5, edgecolor='black')
        pre_bottom += y  


    fontsize = 6
    for x,y in zip(ind, Y2[0]):
        plt.text(x+text_offset, 101, "mode2",horizontalalignment='center', verticalalignment='bottom', fontsize=fontsize)
        ax.annotate(f'{y/100:.1%}', xy=(x-.01,0), xytext=(x+.25,+12),arrowprops=dict(arrowstyle="->",color='C3'), color='C3',fontsize=fontsize)


    ax.set_xticks(np.arange(n), xticks, rotation=0)
    ax.tick_params(axis='x', pad=5)
    ax.set_ylim(0, 100)

    legs = [(x,y) for x,y in zip(h_legends, e_legends)]
    plt.legend(legs, labels, ncol=my_params['ncol'],
                bbox_to_anchor=my_params['anchor'],
                columnspacing=my_params['columnspacing'],
                labelspacing=my_params['labelspacing'],
                handletextpad=my_params['handletextpad'],
                handleheight=my_params['handleheight'],
                handlelength=my_params['handlelength'])


    plt.xlabel(xlabel, labelpad=2)
    plt.ylabel(ylabel, labelpad=2)

    # axes = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(bottom=True, left=True) # x,y轴的刻度线

    ax.spines['bottom'].set_linewidth(params['lines.linewidth'])
    ax.spines['left'].set_linewidth(params['lines.linewidth'])
    ax.spines['right'].set_linewidth(params['lines.linewidth'])
    ax.spines['top'].set_linewidth(params['lines.linewidth'])

    figpath = 'plot.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', pad_inches=0, format='pdf')
    print(figpath, 'is plot.')
    plt.close()


def plot_three_bars(plot_params, data, resolutions, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    bar_width = 0.6
    colors = ['C0', 'C1', 'C2']

    titles = [
        'Full Attention Time',
        'Region Attention Time',
        'Retrieval & Loading Time'
    ]

    ylabels = ['Time (s)', 'Time (s)', 'Time (s)']

    for i, ax in enumerate(axes):
        ax.bar(
            resolutions,
            data[:, i],
            width=bar_width,
            color=colors[i],
            edgecolor='black',
            linewidth=0.8
        )
        ax.set_title(titles[i], fontsize=11)
        ax.set_ylabel(ylabels[i], fontsize=11)
        ax.set_xlabel('Image Resolution', fontsize=11)
        ax.set_ylim(bottom=0)

        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    figpath = 'overhead_bar.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')
    print(figpath, 'is plotted.')
    plt.close()

def plot_resolution_wise_bars(plot_params, data, figpath=None):
    pylab.rcParams.update(plot_params)
    plt.rcParams['pdf.fonttype'] = 42

    resolutions = ['512×512', '1024×1024', '2048×2048']

    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=False)

    bar_width = 0.45

    for i, ax in enumerate(axes):
        full_time = data[i]['full']
        region_time = data[i]['region']
        retrieval_time = data[i]['retrieval']

        # x positions
        x = np.arange(2)

        # Full Attention bar
        ax.bar(
            x[0],
            full_time,
            width=bar_width,
            color='C0',
            edgecolor='black',
            linewidth=0.8,
            label='Full Attention'
        )

        # RegionCache stacked bar
        ax.bar(
            x[1],
            retrieval_time,
            width=bar_width,
            color='C2',
            edgecolor='black',
            linewidth=0.8,
            label='Retrieval & Loading'
        )

        ax.bar(
            x[1],
            region_time,
            width=bar_width,
            bottom=retrieval_time,
            color='C1',
            edgecolor='black',
            linewidth=0.8,
            label='Region Attention'
        )

        ax.set_title(resolutions[i], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['Full', 'RegionCache'], fontsize=10)
        ax.set_ylabel('Time (s)', fontsize=11)

        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='y', labelsize=10)

    # 统一 legend（只放一次）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15)
    )

    plt.tight_layout()
    figpath = 'resolution_wise_overhead.pdf' if not figpath else figpath
    plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')
    print(figpath, 'is plotted.')
    plt.close()

# if __name__ == '__main__':

#   params={
#     'axes.labelsize': '11',
#     'xtick.labelsize':'11',
#     'ytick.labelsize':'11',
#     'lines.linewidth': 1,
#     'legend.fontsize': '11',
#     'figure.figsize' : '4, 2',
#     'legend.loc': 'upper center', #[]"upper right", "upper left"]
#     'legend.frameon': False,
#     'font.family': 'Arial',
#     'font.serif': 'Arial',
#   }


#   Y1 = np.random.randint(0, 101, size=(4, 5))
#   Y2 = np.random.randint(0, 101, size=(4, 5))

#   Y1 = normalized_Y(Y1) * 100
#   Y2 = normalized_Y(Y2) * 100

#   labels = ['stage1', 'stage2', 'stage3', 'stage4']
#   xticks = [f'data{i}' for i in range(5)]
#   xlabel = 'Dataset'
#   ylabel = 'Norm. Execute Time (%)'

#   my_params={
#     'ncol': 2, # 图例列数
#     'anchor': (0.5, 1.48), # 图例位置
#     'columnspacing': 2, # 横向图例间距
#     'labelspacing': 0.5, # 纵向图例间距
#     'handletextpad': 0.8 , # 文字距离
#     'handleheight': 0.7, # 图例高度
#     'handlelength': 2, # 图例宽度

#     'bar_width': 0.25,
#     'colors': ['C0','C1','C2','C3',],
#     'hatchs': ['xx','..','**','++'],
#   }

#   plot_multi_stack_bar(params, my_params, Y1, Y2, labels, xlabel, ylabel, xticks, figpath='multi_stack_bar.pdf')


if __name__ == '__main__':

    params = {
        'axes.labelsize': '11',
        'xtick.labelsize': '10',
        'ytick.labelsize': '10',
        'lines.linewidth': 1,
        'font.family': 'Arial',
        'pdf.fonttype': 42
    }

    data = [
        {'full': 0.292,  'region': 0.141, 'retrieval': 0.047},
        {'full': 1.555,  'region': 0.600, 'retrieval': 0.250},
        {'full': 14.943, 'region': 6.254, 'retrieval': 1.094},
    ]

    plot_resolution_wise_bars(
        plot_params=params,
        data=data,
        figpath='resolution_wise_overhead.pdf'
    )
