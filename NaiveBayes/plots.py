import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_for_macro_weighted_core():
    macro = []
    weighted = []

    ind = np.arange(len(macro))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind - width/2, macro, width,
                    color='red', label='macro')
    rects2 = ax.bar(ind + width/2, weighted, width,
                    color='darkred', label='weighted')

    ax.set_ylabel('scores')
    ax.set_title('F1 score')
    ax.set_xticks(ind)
    ax.set_xticklabels(('normal dist', 'equal freq', 'equal width', 'mdlp'))
    ax.legend(bbox_to_anchor=(0.8, 0.85))

    def autolabel(rects, xpos='center'):
        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.1, 'right': 0.2, 'left': 0.8}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')


    autolabel(rects1, "left")
    autolabel(rects2, "right")

    plt.show()


def cross_val_plot():
    means = [0.59096, 0.59101, 0.60026, 0.59370, 0.59505, 0.593667]
    fig, ax = plt.subplots()

    y1 = np.arange(0, 6)
    sns.barplot(y1, means, palette="Paired")
    ax.set_xticklabels(('10Folds', '10F Stratified', '5Folds', '5F Stratified', '3Folds', '3F Stratified'))

    plt.show()


cross_val_plot()
