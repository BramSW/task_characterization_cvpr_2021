import numpy as np

idx_to_class = {}

with open('/data/cub/CUB_200_2011/image_class_labels.txt') as f:
    for line in f:
        idx, class_i = line.split()
        idx_to_class[int(idx)] = int(class_i)
import csv
task_names_to_ids = {}
task_map = []
taxonomy_path = '/data/cub/CUB_200_2011/taxonomy.txt'

label_to_task_id = {}
with open(taxonomy_path, 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        label = int(row[0])
        label_name = row[5].replace('_', ' ')
        task_name = row[2]

        if task_name not in task_names_to_ids:
            new_task_id = len(task_map)
            task_names_to_ids[task_name] = new_task_id

            task = {
                "task_id": new_task_id,
                "task_name": task_name,
                "class_names": [label_name],
                "class_ids": [0],
                "label_map": {label: 0},
            }
            task_map.append(task)
        else:
            new_task_id = task_names_to_ids[task_name]
            new_label = len(task_map[new_task_id]['label_map'])
            task_map[new_task_id]['class_names'].append(label_name)
            task_map[new_task_id]['class_ids'].append(new_label)
            task_map[new_task_id]['label_map'].update({label: new_label})
        label_to_task_id[int(label)] = int(new_task_id)

idx_to_task_id = {idx:(label_to_task_id[label] if label in label_to_task_id else -1) for idx,label in idx_to_class.items()}
