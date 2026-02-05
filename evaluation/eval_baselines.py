import torch
import torch .nn .functional as F
import numpy as np
import random
from collections import defaultdict
from sklearn .metrics import accuracy_score ,mean_absolute_error ,r2_score
from sklearn .linear_model import LogisticRegression ,Ridge
from sklearn .exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
from config import CONFIG
from time_transf import inverse_transform_time
warnings .filterwarnings ("ignore",module =r"sklearn.*")
def _get_all_test_embeddings (model ,test_tasks_list ,batch_size =64 ):
    all_embeddings =[]
    all_labels =[]
    all_case_ids =[]
    device =next (model .parameters ()).device
    model .eval ()
    try :
        _ =test_tasks_list [0 ][2 ]
    except (IndexError ,TypeError ):
        print ("\n"+"="*50 )
        print ("❌ ERROR in _get_all_test_embeddings (sklearn baseline):")
        print ("Test data does not contain case_ids.")
        print ("This evaluation requires (prefix, label, case_id) tuples.")
        print ("="*50 +"\n")
        return None ,None ,None
    with torch .no_grad ():
        for i in tqdm (range (0 ,len (test_tasks_list ),batch_size ),desc ="Pre-computing test embeddings (for sklearn)"):
            batch_tasks =test_tasks_list [i :i +batch_size ]
            sequences =[t [0 ]for t in batch_tasks ]
            labels =[t [1 ]for t in batch_tasks ]
            case_ids =[t [2 ]for t in batch_tasks ]
            if not sequences :continue
            encoded_batch =model ._process_batch (sequences )
            all_embeddings .append (encoded_batch .cpu ())
            all_labels .extend (labels )
            all_case_ids .extend (case_ids )
    if not all_embeddings :
        return None ,None ,None
    all_embeddings_tensor =torch .cat (all_embeddings ,dim =0 ).to (device )
    all_labels_tensor =torch .as_tensor (all_labels ,device =device )
    return all_embeddings_tensor ,all_labels_tensor
def evaluate_sklearn_baselines (model ,test_tasks ,num_shots_list ,num_test_episodes =100 ):
    strategy =model .strategy
    print (f"\n🧪 Starting evaluation of Scikit-Learn Baselines (feature extraction: '{strategy }')...")
    warnings .filterwarnings ("ignore",category =ConvergenceWarning )
    task_embeddings ={}
    with torch .no_grad ():
        model .eval ()
        for task_type ,task_data in test_tasks .items ():
            if not task_data :
                print (f"Skipping {task_type }: No test data available.")
                continue
            print (f"Pre-computing embeddings for sklearn {task_type }...")
            embeddings ,labels =_get_all_test_embeddings (model ,task_data )
            if embeddings is None :
                return
            task_embeddings [task_type ]=(embeddings ,labels )
    for task_type ,data in task_embeddings .items ():
        print (f"\n--- Baseline task: {task_type } ---")
        all_embeddings ,all_labels =data
        all_labels_np =all_labels .cpu ().numpy ()
        all_embeddings_np =all_embeddings .cpu ().numpy ()
        task_indices =list (range (len (all_labels_np )))
        if task_type =='classification':
            class_dict =defaultdict (list )
            for i ,label in enumerate (all_labels_np ):
                class_dict [label ].append (i )
            class_dict ={c :items for c ,items in class_dict .items ()if len (items )>=max (num_shots_list )+1 }
            if len (class_dict .keys ())<2 :
                print ("Skipping: Not enough classes with sufficient samples.")
                continue
            N_WAYS_TEST =min (len (class_dict .keys ()),7 )
        for k in num_shots_list :
            all_preds ,all_labels_in_test =[],[]
            for _ in range (num_test_episodes ):
                support_indices ,query_indices =[],[]
                if task_type =='classification':
                    eligible_classes =[c for c ,items in class_dict .items ()if len (items )>=k +1 ]
                    if len (eligible_classes )<N_WAYS_TEST :continue
                    episode_classes =random .sample (eligible_classes ,N_WAYS_TEST )
                    for cls in episode_classes :
                        samples_indices =random .sample (class_dict [cls ],k +1 )
                        support_indices .extend (samples_indices [:k ])
                        query_indices .append (samples_indices [k ])
                else :
                    if len (task_indices )<k +1 :continue
                    random .shuffle (task_indices )
                    support_indices =task_indices [:k ]
                    query_indices =task_indices [k :k +1 ]
                if not support_indices or not query_indices :continue
                X_train =all_embeddings_np [support_indices ]
                y_train =all_labels_np [support_indices ]
                X_test =all_embeddings_np [query_indices ]
                y_test =all_labels_np [query_indices ]
                if task_type =='classification':
                    if len (np .unique (y_train ))<2 :continue
                    sk_model =LogisticRegression (max_iter =100 )
                else :
                    sk_model =Ridge ()
                try :
                    sk_model .fit (X_train ,y_train )
                    all_preds .extend (sk_model .predict (X_test ))
                    all_labels_in_test .extend (y_test )
                except ValueError :
                    continue
            if not all_labels_in_test :
                print (f"[{k }-shot] No valid episodes were run.")
                continue
            if task_type =='classification':
                print (f"[{k }-shot] Logistic Regression Accuracy: {accuracy_score (all_labels_in_test ,all_preds ):.4f}")
            else :
                preds =inverse_transform_time (np .array (all_preds ))
                preds [preds <0 ]=0
                labels =inverse_transform_time (np .array (all_labels_in_test ))
                print (
                f"[{k }-shot] Ridge Regression MAE: {mean_absolute_error (labels ,preds ):.4f} | R-squared: {r2_score (labels ,preds ):.4f}")
