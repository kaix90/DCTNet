import numpy as np
import seaborn as sns

def draw_inputgate(gate_np):
    gate_activations_dist = np.average(gate_np.squeeze(), axis=0)
    g = sns.barplot(x=list(range(192)), y=gate_activations_dist)
    g.set_xlabel("Channel Index", fontsize=14)
    g.set_ylabel("Probability of Gates Enabled", fontsize=11)
    # plt.setp(g.get_xticklabels(), rotation=45)
    xticks = g.get_xticks()
    xticks_new = np.concatenate((xticks[:4], xticks[8:12], xticks[16:20], xticks[24:28], xticks[32:36], xticks[64:68],
                                 xticks[72:76], xticks[128:132], xticks[136:140]))
    for n, label in enumerate(g.xaxis.get_ticklabels()):
        if n not in xticks_new:
            label.set_visible(False)
    g.xaxis.set_tick_params(labelsize=6)
    np.save('input_gate.npy', gate_activations_dist)
    g.figure.savefig('inputgate.png')
    g.figure.savefig('inputgate.pdf')


def zigZag(arr):
    rows, columns = len(arr), len(arr[0])
    result = [[] for i in range(rows + columns - 1)]

    for i in range(rows):
        for j in range(columns):
            sum = i + j
            if (sum % 2 == 0):

                # add at beginning
                result[sum].insert(0, arr[i][j])
            else:

                # add at end of the list
                result[sum].append(arr[i][j])
    return result

def draw_from_npy(filename):
    import matplotlib.pyplot as plt
    gate_activations_dist = np.load(filename)
    y = gate_activations_dist[:64].reshape((8, 8))
    cb = gate_activations_dist[64:128].reshape((8, 8))
    cr = gate_activations_dist[128:].reshape((8, 8))

    plt.figure(1, figsize = (32, 32))
    plt.subplot(411)
    ax = sns.heatmap(y, linewidth=0.5, cmap="OrRd", square=True)

    plt.subplot(412)
    ax = sns.heatmap(cb, linewidth=0.5, cmap="OrRd", square=True)

    plt.subplot(413)
    ax = sns.heatmap(cr, linewidth=0.5, cmap="OrRd", square=True)

    plt.subplot(414)
    list_a = list(np.arange(64))
    list_b = [x for sublist in zigZag(np.asarray(list_a).reshape((8, 8))) for x in sublist]
    list_c = [list_b.index(m) for m in list_a]
    ax = sns.heatmap(np.asarray(list_c).reshape((8, 8)), linewidth=0.5, cmap="OrRd", square=True, annot=True, annot_kws={"size": 18})
    # ax = sns.heatmap(np.arange(64).reshape((8, 8)), linewidth=0.5, cmap="OrRd", square=True, annot=True, annot_kws={"size": 18})
    # plt.show()
    plt.savefig('heatmap.svg')
    print('heatmap saved.')

if __name__ == '__main__':
    # main()
    draw_from_npy('/mnt/kai/work/code/dctDet/input_gate.npy')