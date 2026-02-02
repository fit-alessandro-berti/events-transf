import pandas as pd
import random
import numpy as np
from collections import defaultdict
from time_transf import transform_time
def get_task_data (log ,task_type ,max_seq_len =10 ):
    tasks =[]
    if not log :return tasks
    for trace in log :
        if len (trace )<3 :continue
        case_id =trace [0 ]['case_id']
        for i in range (1 ,len (trace )-1 ):
            prefix =trace [:i +1 ]
            if len (prefix )>max_seq_len :prefix =prefix [-max_seq_len :]
            next_event_activity_id =trace [i +1 ]['activity_id']
            if task_type =='classification':
                if next_event_activity_id is not None :
                    tasks .append ((prefix ,next_event_activity_id ,case_id ))
            elif task_type =='regression':
                remaining_time =(trace [-1 ]['timestamp']-prefix [-1 ]['timestamp'])/3600.0
                tasks .append ((prefix ,transform_time (remaining_time ),case_id ))
    return tasks
def create_episode (task_pool ,num_shots_range ,num_queries_per_class ,num_ways_range =(2 ,5 ),shuffle_labels =False ):
    class_dict =defaultdict (list )
    for seq ,label ,case_id in task_pool :
        class_dict [label ].append ((seq ,label ))
    num_ways =random .randint (num_ways_range [0 ],num_ways_range [1 ])
    num_shots =random .randint (num_shots_range [0 ],num_shots_range [1 ])
    available_classes =[c for c ,items in class_dict .items ()if len (items )>=num_shots +num_queries_per_class ]
    if len (available_classes )<num_ways :return None
    episode_classes =random .sample (available_classes ,num_ways )
    label_map ={}
    if shuffle_labels :
        shuffled_classes =random .sample (episode_classes ,len (episode_classes ))
        label_map ={original :shuffled for original ,shuffled in zip (episode_classes ,shuffled_classes )}
    support_set ,query_set =[],[]
    for cls in episode_classes :
        mapped_label =label_map .get (cls ,cls )
        samples =random .sample (class_dict [cls ],num_shots +num_queries_per_class )
        for s in samples [:num_shots ]:
            support_set .append ((s [0 ],mapped_label ))
        for s in samples [num_shots :]:
            query_set .append ((s [0 ],mapped_label ))
    random .shuffle (support_set )
    random .shuffle (query_set )
    return support_set ,query_set
