#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


plt.rcParams['figure.dpi'] = 600

path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_vec_out.json'
path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_vec_rlebits_4/'

# path_res = '/Users/danieldonenfeld/Developer/taco/rle_test/bench_mat_out.json'
# path_out = '/Users/danieldonenfeld/Developer/taco/rle_test/plots_mat/'


def split_name_str(s):
    res = s.split(':')
    return (res[0], res[1])


def to_str(lst):
    s = ""
    for e in lst:
        s += str(e) + ","
    return s[:-1]


def parse_name(name):
    name = name.split('/')
    _, size = split_name_str(name[1])
    vals_l = 0
    _, vals_u = split_name_str(name[2])
    _, rle_l = split_name_str(name[3])
    _, rle_u = split_name_str(name[4])
    _, rle_bits = split_name_str(name[5])
    elide_overflow_checks = bool(int(name[6]))
    return (int(size), int(vals_l), int(vals_u), int(rle_l), int(rle_u), int(rle_bits), elide_overflow_checks)


def process_benchmark(b, d, s, sg):
    size, vals_l, vals_u, rle_l, rle_u, rle_bits, elide_overflow = parse_name(b['name'])

    s.add(rle_bits)
    sg.add(to_str([vals_l, vals_u, rle_l, rle_u]))

    d_key = to_str([vals_l, vals_u, rle_l, rle_u, rle_bits, elide_overflow])
    xs, ys = d.setdefault(d_key, ([], []))
    xs.append(size)
    ys.append(b['cpu_time'])


def process_all():
    with open(path_res) as f:
        data = json.load(f)

    d = dict()
    s = set()
    sg = set()

    for b in data['benchmarks']:
        process_benchmark(b, d, s, sg)
    
    rle_bits_l = list([int(i) for i in s])
    rle_bits_l.sort()
    return (d,rle_bits_l,sg)


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


def create_data_frame(xs, ys_lists, names):
    columns = ['Vector Length'] + names

    ys_groups = list(zip(*ys_lists))
    ys_groups = list(map(lambda x: [x[0]/i for i in x], ys_groups))
    ys_groups = list(map(lambda x: ["{:1.0e}".format(int(xs[x[0]]))]+list(x[1]), zip(range(len(ys_groups)), ys_groups)))

    df = pd.DataFrame(ys_groups, columns=columns)
    return df

def create_bar_charts():
    d,rle_bits_l,sg = process_all()


    for g in sg:
        # Each iteration should produce one graph
        print(g)

        fig, ax = plt.subplots()

        data = []
        for bits in rle_bits_l:
            g_key = g + "," + str(bits) + ",False"
            xs, ys = d[g_key]
            name = "rle " + str(bits) + " bits" if bits > 0 else "dense"
            data.append((xs, ys, name))
            if (bits > 8):
                name += "[NOC]"
                g_key = g + "," + str(bits) + ",True"
                xs, ys = d[g_key]
                data.append((xs, ys, name))


        g = g.split(',')

        xs_all = data[0][0]
        for xs, ys, name in data:
            if not (xs == xs_all):
                print("ERROR! x values not same")

        ys_lists = [ys for _,ys,_ in data]
        names = [name for _,_,name in data]
        # for xs, ys in data:
        #     ys_lists.append(ys)

        df = create_data_frame(xs_all, ys_lists, names)
                
        # plot grouped bar chart
        df.plot(x='Vector Length',
                kind='bar',
                stacked=False,
                title="Values: [" + g[0] + "," + g[1] +
                    "], RLE: [" + g[2] + "," + g[3] + "]")

        plt.tight_layout()
        plt.savefig(path_out+"Values_" + g[0] + "_" + g[1] + "_rle_" + g[2] + "_"
                    + g[3] + ".png")
        plt.close()


def create_scatter():
    with open(path_res) as f:
        data = json.load(f)

    l = []

    for b in data['benchmarks']:
        size, vals_l, vals_u, rle_l, rle_u, rle_bits, elide_overflow = parse_name(b['name'])

        total_vals = int(b['t0_vals_size_total']) + int(b['t1_vals_size_total'])
        time = float(b['cpu_time'])
        throughput = total_vals/time

        l.append([rle_bits, throughput, math.log10(size)])

    df = pd.DataFrame(l,
                columns=['bits', 'throughput', 'size'])
    ax1 = df.plot.scatter(x='bits',
                    y='throughput',
                    c='size',
                    colormap='viridis',
                    s=3)
    plt.tight_layout()
    plt.savefig(path_out+"scatter.png")
    plt.close()


if __name__ == "__main__":
    create_bar_charts()
    create_scatter()