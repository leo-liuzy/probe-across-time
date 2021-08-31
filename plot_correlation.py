import os
import sys
import csv
import json
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
CHECKPOINTS = {0, 3200, 6400, 20000, 40000, 80000, 160000, 320000, 420000, 520000, 640000, 760000, 880000, 1000000}
ticks = []
data = []
chkpt_header = "checkpoint (in update steps)"
summary_root = "summaries"
scenario = "ALL (161GB)"
# def read_checkpoints():

# load LKT
probe_name = "LKT"
for test in os.listdir(f"{summary_root}/{probe_name}"):
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    tick = f"{probe_name}-{test}"
    ticks.append(tick)
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    metrics = table['metric'][table[chkpt_header].isin(CHECKPOINTS)].to_list()
    assert len(metrics) == len(CHECKPOINTS)
    data.append(metrics)

# load BLiMP
probe_name = "BLiMP"
for test in os.listdir(f"{summary_root}/{probe_name}"):
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    tick = f"{probe_name}-{test}"
    ticks.append(tick)
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    metrics = table['metric'][table[chkpt_header].isin(CHECKPOINTS)].to_list()
    assert len(metrics) == len(CHECKPOINTS)
    data.append(metrics)
# load LAMA
print()
probe_name = "LAMA"
K = 1
assert K in [1, 5, 10]
for test in os.listdir(f"{summary_root}/{probe_name}"):
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    table.loc[:, "metric"] = table.loc[:, "metric"].apply(lambda x: eval(x)[K])
    sub_table = table[table[chkpt_header].isin(CHECKPOINTS)]
    # sub_table.loc[:, "metric"] = sub_table.loc[:, "metric"].apply(lambda x: eval(x)[K])
    for relation_name, relation_sub_table in sub_table.groupby("relation_type"):
        ticks.append(f"{probe_name}-{test}-{relation_name}")
        metrics = relation_sub_table['metric'].tolist()
        data.append(metrics)
# load CAT
probe_name = "CAT"
for test in os.listdir(f"{summary_root}/{probe_name}"):
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    tick = f"{probe_name}-{test}"
    ticks.append(tick)
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    metrics = table['metric'][table[chkpt_header].isin(CHECKPOINTS)].to_list()
    assert len(metrics) == len(CHECKPOINTS)
    data.append(metrics)

# load oLMpics
probe_name = "oLMpics"
for test in os.listdir(f"{summary_root}/{probe_name}"):
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    tick = f"{probe_name}-{test}"
    ticks.append(tick)
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    metrics = table['metric'][table[chkpt_header].isin(CHECKPOINTS)].to_list()
    assert len(metrics) == len(CHECKPOINTS)
    data.append(metrics)
print()
# load Finetune
for test in ["CoLA", "MRPC", "SST-2", "WNLI", "WSC"]:
    if not os.path.isdir(f"{summary_root}/{probe_name}/{test}"):
        continue
    tick = f"{probe_name}-{test}"
    ticks.append(tick)
    table = pandas.read_csv(f"{summary_root}/{probe_name}/{test}/{scenario}.csv", index_col=False)
    table.sort_values(by=[chkpt_header], inplace=True)
    metrics = table['metric'][table[chkpt_header].isin(CHECKPOINTS)].to_list()
    assert len(metrics) == len(CHECKPOINTS)
    data.append(metrics)
