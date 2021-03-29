#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np

path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_vec_out.json'
path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_vec_rlebits_1/'

# path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_mat_out.json'
# path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_mat/'


def split_name_str(s):
    res = s.split(':')
    return (res[0], res[1])


def to_str(lst):
    s = ""
    for e in lst:
        s += str(e) + ","
    return s


def process_benchmark(b, d, s, sg):
    name = b['name']
    name = name.split('/')

    _, size = split_name_str(name[1])
    vals_l = 0
    _, vals_u = split_name_str(name[2])
    _, rle_l = split_name_str(name[3])
    _, rle_u = split_name_str(name[4])
    _, rle_bits = split_name_str(name[5])

    s.add(rle_bits)
    sg.add(to_str([vals_l, vals_u, rle_l, rle_u]))

    d_key = to_str([vals_l, vals_u, rle_l, rle_u, rle_bits])
    xs, ys = d.setdefault(d_key, ([], []))
    xs.append(size)
    ys.append(b['cpu_time'])


def add_bars(ax, xs_all, ys_lists, ind, width):
    rects = []
    i = 0
    n = len(ys_lists)
    k = 0
    for ys in ys_lists:
        lbl = "dense" if i == 0 else "rle " + str(i) + " bits"
        rects.append(ax.bar(ind - ((n-k)*width/n) +
                            width/2, ys, width/n, label=lbl))
        k += 1
        i = 8 if i == 0 else i*2
    return rects


with open(path_res) as f:
    data = json.load(f)

d = dict()
s = set()
sg = set()

for b in data['benchmarks']:
    process_benchmark(b, d, s, sg)

rle_bits_l = list(s)
rle_bits_l.sort()

for g in sg:
    # Each iteration should produce one graph
    print(g)

    fig, ax = plt.subplots()

    data = []
    for bits in rle_bits_l:
        g_key = g + str(bits) + ","
        xs, ys = d[g_key]
        data.append((xs, ys))

    g = g.split(',')

    xs_all = data[0][0]
    for xs, ys in data:
        if not (xs == xs_all):
            print("ERROR! x values not same")

    ys_lists = []
    for xs, ys in data:
        ys_lists.append(ys)

    N = len(xs_all)
    ind = np.arange(N)
    width = 0.6
    num_bars = len(rle_bits_l)
    labels = list(map(lambda x: "{:.0e}".format(int(x)), xs_all))

    rects = add_bars(ax, xs_all, ys_lists, ind, width)
    ax.set_yscale('log')
    ax.set_xlabel('Vector Length')
    ax.set_ylabel('Time (us)')
    ax.set_title("Values: [" + g[0] + "," + g[1] +
                 "], RLE: [" + g[2] + "," + g[3] + "]")
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.savefig(path_out+"Values_" + g[0] + "_" + g[1] + "__rle_" + g[2] + "_"
                + g[3] + ".png")
    plt.close()


# def autolabel(rects_l, rects_r, data_l, data_r):
#     data_zip = zip(data_l, data_r)
#     labels = list(map(lambda x: "{:10.2f}".format(x[0]/x[1]), data_zip))

#     for l, r, v in zip(rects_l, rects_r, labels):
#         height = max(l.get_height(), r.get_height())
#         ax.annotate('{}'.format(v),
#                     # xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xy=(r.get_x() - r.get_width()/4, height),
#                     xytext=(0, 2),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# for k, v in d.items():
#     print(k)

#     fig, ax = plt.subplots()

#     dense_data_x, dense_data_y, rle_data_x, rle_data_y = v

#     if not (dense_data_x == rle_data_x):
#         print("ERROR! x values not same")

#     N = len(dense_data_x)
#     ind = np.arange(N)
#     width = 0.4

#     labels = list(map(lambda x: "{:.0e}".format(int(x)), dense_data_x))

#     rects_dense = ax.bar(ind - width/2, dense_data_y, width, label="dense")
#     rects_rle = ax.bar(ind + width/2, rle_data_y, width, label='RLE')
#     ax.set_yscale('log')
#     # ax.set_xlabel('Vector Length')
#     ax.set_xlabel('Matrix Size')
#     ax.set_ylabel('Time (ms)')
#     ax.set_title(k)
#     ax.set_xticks(ind)
#     ax.set_xticklabels(labels)
#     ax.legend(loc="best")

#     autolabel(rects_dense, rects_rle, dense_data_y, rle_data_y)

#     fig.tight_layout()
#     plt.savefig(path_out+k+".png")
#     plt.close()
