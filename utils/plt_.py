import os, pandas as pd
import matplotlib.pyplot as plt

def group_statistics(df, dump=''):
    gg = df.groupby('cat')
    lst = []
    for a, g in gg:
        lst.append( [a, g.shape[0], g.time.sum(), g.time.mean(), g.time.std()] )
    dd = pd.DataFrame(np.array(lst, dtype='object'), columns=['cat','#Rows', 'sum', 'mean', 'std'])
    if dump != '':
        dd.to_excel(dump+'.xlsx', index=False)

def read(f):
    # path = 'skill_%d model'%skill
    df = pd.read_csv(f)
    df.rename(columns={' ':'time_points'}, inplace=True)
    df.set_index('time_points', inplace=True)
    return df.T

def plt_(skill, trial):
    path = 'skill_%d model'%skill
    model_min = read('%s/%s.csv'%(path, 'min'))
    model_max = read('%s/%s.csv'%(path, 'max'))
    model_std = read('%s/%s.csv'%(path, 'std'))
    model_avg = read('%s/%s.csv'%(path, 'mean'))

    t = read(trial)

    if not (model_min.shape == model_max.shape == model_std.shape == model_avg.shape == t.shape):
        print('Have different shapes')
        print('model_min.shape ----------', model_min.shape)
        print('model_max.shape ----------', model_max.shape)
        print('model_std.shape ----------', model_std.shape)
        print('model_avg.shape ----------', model_avg.shape)
        print('trial.shape ----------', t.shape)
        return

    try:
        fig_path = trial[:-4] + ' - figs'
        os.mkdir(fig_path)
    except:
        ...
    data_x = range(1, len(model_max) + 1)
    line_style = ['dashdot','dotted','dashed'][1]
    for col in t.columns:
        #     plt.plot(data_x, final, linestyle='dashed', color='blue', label="mean")
        plt.plot(data_x, model_max[col], linestyle=line_style, color='green', label="Max")
        plt.plot(data_x, model_avg[col] + model_std[col], linestyle=line_style, color='black', label="Std+")
        plt.plot(data_x, model_avg[col], linestyle=line_style, color='pink', label="Mean")
        plt.plot(data_x, model_avg[col] - model_std[col], linestyle=line_style, color='orange', label="Std-")
        plt.plot(data_x, model_min[col], linestyle=line_style, color='blue', label="Min")

        file_name = "%s" % col.replace('/', '_')     ############ p
        plt.tick_params(labelsize=15)                ############ p
        plt.legend(loc='best', fontsize=13)          ############ p
        plt.grid()                                   ############ p


        ###########
        plt.savefig('%s/%s without_red.pdf' % (fig_path, file_name), bbox_inches='tight')
        ###########

        plt.plot(data_x, t[col], color='red', label="Trail")
        plt.legend(loc='best', fontsize=13)
        plt.savefig('%s/%s with_red.pdf' % (fig_path, file_name), bbox_inches='tight')
        plt.clf()