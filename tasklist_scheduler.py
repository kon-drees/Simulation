import pandas as pd
import numpy as np


task_data = pd.read_csv('t-1-4.txt', delim_whitespace=True)


# calculates the max duration in an even distributed scheduler
def even_list_scheduler(dataframe, number_tasks, ressource):
    if dataframe.shape[0] < number_tasks:
        number_tasks = dataframe.shape[0]
    res_list = np.zeros(ressource)
    for task in range(number_tasks):
        current_task = dataframe.loc[task]
        res_list[task % ressource] += current_task['Runtime']
    return res_list.max()


# calculates the max duration in an optimized distributed scheduler
def list_scheduler(dataframe, number_tasks, ressource):
    if dataframe.shape[0] < number_tasks:
        number_tasks = dataframe.shape[0]
    res_list = np.zeros(ressource)
    for task in range(number_tasks):
        current_task = dataframe.loc[task]
        min_nr = np.amin(res_list)
        for i in range(ressource):
            if res_list[i] == min_nr:
                res_list[i] += current_task['Runtime']
                break
    return res_list.max()


print(list_scheduler(task_data, 1000, 2))
print(list_scheduler(task_data, 1000, 4))
print(list_scheduler(task_data, 1000, 8))
print(list_scheduler(task_data, 1000, 16))
print(even_list_scheduler(task_data, 1000, 2))
print(even_list_scheduler(task_data, 1000, 4))
print(even_list_scheduler(task_data, 1000, 8))
print(even_list_scheduler(task_data, 1000, 16))

data = {'Name': ['imgread', 'grayscale left', 'grayscale right', 'sobel left', 'sobel right', 'SDE', 'imgwrite'], 'Runtime': [1, 2, 2, 6, 6, 10, 1]}
image_processing_tasks = pd.DataFrame(data)
for _ in range(999):
    image_processing_tasks = image_processing_tasks.append(pd.DataFrame(data), ignore_index=True)

print(list_scheduler(image_processing_tasks, len(image_processing_tasks), 4))

