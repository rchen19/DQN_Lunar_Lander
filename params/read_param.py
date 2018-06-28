import numpy as np
import pandas as pd
import os

task_list = pd.read_csv(os.path.join("task.csv"), index_col=0)

for index in task_list.index:
    params = dict()
    for column in task_list.columns:
        if column == "tau" and np.isnan(task_list[column][index]):
            params[column] = None
        else:
            params[column] = task_list[column][index]


print(len(task_list.index))
print(len(task_list.columns))
print(sorted(task_list.index))