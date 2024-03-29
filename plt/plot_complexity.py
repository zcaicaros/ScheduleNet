import numpy as np
import matplotlib.pyplot as plt


fixed = 'j=40'  # 'j=40', 'm=10'

datas = [
    'RL-GNN_complexity_{}_reimplement.npy'.format(fixed),
    'L2S_complexity_{}_[500].npy'.format(fixed),
    'ScheduleNet_complexity_{}_reimplement.npy'.format(fixed),
]

times_for_plot = []

for data in datas:
    times_for_plot.append(np.load(data).reshape(-1))
x_labels = [str(10+5*i) for i in range(times_for_plot[1].shape[0])]

# plot parameters
x_label_scale = 15
y_label_scale = 15
anchor_text_size = 15
title_size = 15
show = False
save = True
save_file_type = '.pdf'


obj0 = times_for_plot[0]  # RL-GNN Reimplement
obj1 = times_for_plot[1]  # ours-500
obj2 = times_for_plot[2]  # ScheduleNet

# plotting...
plt.figure(figsize=(5.3, 5))
if fixed == 'm=5':
    plt.xlabel('Number of jobs {}'.format(r'$n$'), {'size': x_label_scale})
elif fixed == 'j=30':
    plt.xlabel('Number of machines {}'.format(r'$m$'), {'size': x_label_scale})

plt.ylabel('Average run time (seconds)', {'size': y_label_scale})
if fixed == 'j=40':
    plt.xlabel('Number of machines', {'size': x_label_scale})
if fixed == 'm=10':
    plt.xlabel('Number of jobs', {'size': x_label_scale})
plt.grid()
x = np.array(x_labels)
plt.plot(x, obj0, color='tab:red', marker="s", label='RL-GNN')
plt.plot(x, obj2, color='tab:brown', marker="*", label='ScheduleNet')
plt.plot(x, obj1, color='tab:blue', marker="v", label='Ours-500')

plt.tight_layout()
plt.legend(fontsize=anchor_text_size)
if save:
    plt.savefig('./complexity_analysis_{}{}'.format(fixed, save_file_type))
if show:
    plt.show()