#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_vec_out.json'
# path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_vec/'

path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_mat_out.json'
path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_mat/'

with open(path_res) as f:
  data = json.load(f)

d = dict()
  
for b in data['benchmarks']:
    name = b['name']
    name = name.split('/')

    case   = name[0]
    size   = name[1]
    vals_l = 0
    vals_u = name[2]
    rle_l  = name[3]
    rle_u  = name[4]

    d_key = "values: [" + str(vals_l) + "," + str(vals_u) +"]"+ ", run lengths: [" + str(rle_l) + "," + str(rle_u) +"]"

    if 'dense' in case:
        dense_data_x,dense_data_y,_,_ = d.setdefault(d_key, ([],[],[],[]))
        dense_data_x.append(size)
        dense_data_y.append(b['cpu_time'] / 1e6) # convert to ms
    else:
        _,_,rle_data_x,rle_data_y = d.setdefault(d_key, ([],[],[],[]))
        rle_data_x.append(size)
        rle_data_y.append(b['cpu_time'] / 1e6) # convert to ms


def autolabel(rects_l, rects_r, data_l, data_r):
    data_zip = zip(data_l, data_r)
    labels = list(map(lambda x: "{:10.2f}".format(x[0]/x[1]), data_zip))

    for l,r,v in zip(rects_l, rects_r, labels):
        height = max(l.get_height(), r.get_height())
        ax.annotate('{}'.format(v),
                    # xy=(rect.get_x() + rect.get_width() / 2, height),
                    xy=(r.get_x() - r.get_width()/4, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
for k,v in d.items():
    print(k)

    fig, ax = plt.subplots()
    
    dense_data_x,dense_data_y,rle_data_x,rle_data_y = v

    if not (dense_data_x == rle_data_x):
        print("ERROR! x values not same")
    
    N = len(dense_data_x)
    ind = np.arange(N)
    width = 0.4

    labels = list(map(lambda x: "{:.0e}".format(int(x)), dense_data_x))
    
    rects_dense = ax.bar(ind - width/2, dense_data_y, width, label="dense")
    rects_rle   = ax.bar(ind + width/2, rle_data_y, width, label='RLE')
    ax.set_yscale('log')
    # ax.set_xlabel('Vector Length')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title(k)
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.legend(loc="best")

    autolabel(rects_dense, rects_rle, dense_data_y, rle_data_y)
    
    fig.tight_layout()
    plt.savefig(path_out+k+".png")
    plt.close()
