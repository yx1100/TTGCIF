# coding=utf-8
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def shuffle(x):
    return float(x.strip())


domains = ['Arts',
           'CD',
           ]

colors = ['#B8860B',
          '#98FB98'
          ]

for index in range(len(domains)):
    domain = domains[index]
    color = colors[index]
    file = open("./LOSS/%s_LOSS.txt" % domain, "r")
    art_lines = file.readlines()
    file.close()
    art_lines = map(shuffle, art_lines)
    art_lines = np.array(art_lines)
    x_art = range(1, art_lines.shape[0] + 1)
    y_art = art_lines
    plt.plot(x_art, y_art, color=colors[index], linewidth=2.0, linestyle='-.', label='%s -> Toy' % domain)

plt.ylabel("MMD Loss")
plt.xlabel("Training Step")
# plt.fill_between(x,y, alpha=0.15)
plt.grid(alpha=0.3, linestyle='-')
plt.legend()
plt.show()
