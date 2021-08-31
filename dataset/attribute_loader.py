import re
from collections import defaultdict
from pprint import pprint
from .image_index_to_task_map import idx_to_task_id

attr_family_to_attr_indices_and_names = defaultdict(list)
with open('/data/cub/CUB_200_2011/attributes/attributes.txt') as f:
    for line in f:
        attribute_family = re.search('(?<=has_).*(?=::)', line).group(0)
        attr_ind = int(line.split()[0])
        attr_name = line.split('::')[1].strip()
        attr_family_to_attr_indices_and_names[attribute_family].append( (attr_ind, attr_name) )

image_and_attr_inds_to_val = {}
with open('/data/cub/CUB_200_2011/attributes/image_attribute_labels.txt') as f:
    for line in f:
        try:
            image_ind, attr_ind, val, _, _ = line.split()
        except:
            # 2275 and 9364 have extra 0 before time
            try:
                image_ind, attr_ind, val, _, extra, _ = line.split()
                assert extra=='0' and image_ind in ('2275', '9364'), "Two images have extra 0 entry in file, they are 2275 and 9364"
            except:
                raise
        image_and_attr_inds_to_val[(int(image_ind), int(attr_ind))] = int(val)

def attr_family_to_attr_indices(family):
    return [tup[0] for tup in attr_family_to_attr_indices_and_names[family]]

def attr_family_and_image_index_to_attr_label(family, image_ind, adjust_for_task_id=True):
    attr_indices = attr_family_to_attr_indices(family)
    attr_index_to_label_map = {i:attr_i for i,attr_i in enumerate(attr_indices)}
    # Now need to harvest labels for ind
    indiv_attr_labels = [image_and_attr_inds_to_val[(image_ind,attr_i)]  for attr_i in attr_indices]
    # Note that above is list in order of attr_index_to_label_map
    attr_label = indiv_attr_labels.index(1) if (sum(indiv_attr_labels)==1) else None
    if attr_label is not None and adjust_for_task_id:
        attr_label += 100 * idx_to_task_id[image_ind]
    # print("Family: {} / Image Index: {} / Attr. Derived Label: {}".format(family, image_ind, attr_label))
    return attr_label
