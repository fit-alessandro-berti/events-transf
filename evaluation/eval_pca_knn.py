import torch
import torch .nn .functional as F
import random
import numpy as np
from sklearn .metrics import accuracy_score ,mean_absolute_error ,r2_score
from sklearn .decomposition import PCA
from scipy .stats import mode as scipy_mode
from tqdm import tqdm
import warnings
from time_transf import inverse_transform_time
from utils .retrieval_utils import find_knn_indices
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
        print ("❌ ERROR in _get_all_test_embeddings (PCA baseline):")
        print ("Test data does not contain case_ids.")
        print ("This evaluation requires (prefix, label, case_id) tuples.")
        print ("="*50 +"\n")
        return None ,None ,None
    with torch .no_grad ():
        for i in tqdm (range (0 ,len (test_tasks_list ),batch_size ),desc ="Pre-computing test embeddings (for PCA)"):
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
    all_case_ids_array =np .array (all_case_ids )
    return all_embeddings_tensor ,all_labels_tensor ,all_case_ids_array
def evaluate_pca_knn (model ,test_tasks ,k_list ,num_test_queries =200 ):
    print ("\n🔬 Starting PCA + k-NN Baseline Evaluation...")
    model .eval ()
    task_embeddings ={}
    for task_type ,task_data in test_tasks .items ():
        if not task_data :
            print (f"Skipping {task_type }: No test data available.")
            continue
        embeddings ,labels ,case_ids =_get_all_test_embeddings (model ,task_data )
        if embeddings is None :return
        print (f"  - Applying PCA to {embeddings .shape [0 ]} embeddings...")
        embeddings_np =embeddings .cpu ().numpy ()
        n_samples ,n_features =embeddings_np .shape
        n_components =min (128 ,n_samples ,n_features )
        if n_components <1 :
            print (f"Skipping {task_type }: Not enough samples/features for PCA.")
            continue
        with warnings .catch_warnings ():
            warnings .simplefilter ("ignore",UserWarning )
            pca =PCA (n_components =n_components )
            pca_embeddings =pca .fit_transform (embeddings_np )
        print (f"  - Reduced dimension from {n_features } to {n_components }")
        pca_embeddings_tensor =torch .from_numpy (pca_embeddings ).to (embeddings .device )
        pca_embeddings_norm =F .normalize (pca_embeddings_tensor ,p =2 ,dim =1 )
        task_embeddings [task_type ]=(pca_embeddings_norm ,labels ,case_ids )
    for task_type ,(all_embeddings ,all_labels ,all_case_ids )in task_embeddings .items ():
        print (f"\n--- Evaluating PCA-kNN task: {task_type } ---")
        num_total_samples =all_embeddings .shape [0 ]
        if num_total_samples <2 :
            print ("Skipping: Not enough samples to evaluate.")
            continue
        num_queries =min (num_test_queries ,num_total_samples )
        query_indices =random .sample (range (num_total_samples ),num_queries )
        for k in k_list :
            if k >=num_total_samples :
                print (f"Skipping [k={k }]: k is larger than total samples.")
                continue
            all_preds ,all_true_labels =[],[]
            for query_idx in query_indices :
                query_embedding =all_embeddings [query_idx :query_idx +1 ]
                query_label =all_labels [query_idx ]
                query_case_id =all_case_ids [query_idx ]
                same_case_indices_np =np .where (all_case_ids ==query_case_id )[0 ]
                mask_tensor =torch .from_numpy (same_case_indices_np ).to (query_embedding .device )
                top_k_indices =find_knn_indices (
                query_embedding ,
                all_embeddings ,
                k =k ,
                indices_to_mask =mask_tensor
                )
                if top_k_indices .numel ()==0 :
                    continue
                support_labels =all_labels [top_k_indices ]
                with torch .no_grad ():
                    if task_type =='classification':
                        support_labels_np =support_labels .cpu ().numpy ()
                        pred_label =scipy_mode (support_labels_np ,keepdims =True ).mode [0 ]
                        all_preds .append (pred_label )
                        all_true_labels .append (query_label .item ())
                    else :
                        pred_value =support_labels .float ().mean ().item ()
                        all_preds .append (pred_value )
                        all_true_labels .append (query_label .item ())
            if not all_true_labels :continue
            if task_type =='classification':
                print (
                f"[{k }-NN] PCA-kNN Accuracy: {accuracy_score (all_true_labels ,all_preds ):.4f} (on {len (all_true_labels )} queries)")
            else :
                preds_np =np .array (all_preds )
                labels_np =np .array (all_true_labels )
                preds =inverse_transform_time (preds_np );
                preds [preds <0 ]=0
                labels =inverse_transform_time (labels_np )
                print (
                f"[{k }-NN] PCA-kNN MAE: {mean_absolute_error (labels ,preds ):.4f} | R-squared: {r2_score (labels ,preds ):.4f}")
