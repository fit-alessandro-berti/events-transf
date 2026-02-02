import random
import torch
import torch .nn .functional as F
import numpy as np
from utils .retrieval_utils import find_knn_indices
def run_retrieval_step (model ,task_data_pool ,task_type ,config ):
    progress_bar_task =f"retrieval_{task_type }"
    retrieval_k_train =config .get ('retrieval_train_k',5 )
    retrieval_batch_size =config .get ('retrieval_train_batch_size',64 )
    if len (task_data_pool )<retrieval_batch_size :
        return None ,progress_bar_task
    batch_tasks_raw =random .sample (task_data_pool ,retrieval_batch_size )
    batch_prefixes =[t [0 ]for t in batch_tasks_raw ]
    batch_labels =np .array ([t [1 ]for t in batch_tasks_raw ])
    batch_case_ids =np .array ([t [2 ]for t in batch_tasks_raw ])
    all_embeddings =model ._process_batch (batch_prefixes )
    device =all_embeddings .device
    with torch .no_grad ():
        all_embeddings_norm =F .normalize (all_embeddings ,p =2 ,dim =1 )
    total_loss_for_batch =0.0
    queries_processed =0
    for i in range (retrieval_batch_size ):
        query_label =batch_labels [i ]
        query_case_id =batch_case_ids [i ]
        query_embedding =all_embeddings [i :i +1 ]
        positive_mask =(batch_labels ==query_label )&(batch_case_ids !=query_case_id )
        positive_indices =np .where (positive_mask )[0 ]
        if len (positive_indices )==0 :
            continue
        chosen_positive_idx =random .choice (positive_indices )
        with torch .no_grad ():
            query_embedding_norm =all_embeddings_norm [i :i +1 ]
            same_case_mask =(batch_case_ids ==query_case_id )
            same_case_indices =np .where (same_case_mask )[0 ]
            indices_to_mask_np =np .append (same_case_indices ,chosen_positive_idx )
            mask_tensor =torch .from_numpy (indices_to_mask_np ).to (device )
            num_neighbors_to_find =retrieval_k_train -1
        if num_neighbors_to_find <0 :
            neighbor_indices =torch .tensor ([],dtype =torch .long ,device =device )
        else :
            neighbor_indices =find_knn_indices (
            query_embedding_norm ,
            all_embeddings_norm ,
            k =num_neighbors_to_find ,
            indices_to_mask =mask_tensor
            )
        support_indices =torch .cat ([neighbor_indices ,torch .tensor ([chosen_positive_idx ],device =device )])
        support_embeddings =all_embeddings [support_indices ]
        support_labels_list =batch_labels [support_indices .cpu ().numpy ()]
        if task_type =='classification':
            support_labels_tensor =torch .LongTensor (support_labels_list ).to (device )
            logits ,proto_classes ,_ =model .proto_head .forward_classification (
            support_embeddings ,support_labels_tensor ,query_embedding
            )
            if logits is None :continue
            label_map ={orig .item ():new for new ,orig in enumerate (proto_classes )}
            mapped_label =torch .tensor ([label_map .get (query_label ,-100 )],device =device ,dtype =torch .long )
            if mapped_label .item ()==-100 :
                continue
            loss =F .cross_entropy (logits ,mapped_label ,label_smoothing =0.05 )
        else :
            support_labels_tensor =torch .as_tensor (support_labels_list ,dtype =torch .float32 ,device =device )
            query_label_tensor =torch .as_tensor ([query_label ],dtype =torch .float32 ,device =device )
            prediction ,_ =model .proto_head .forward_regression (
            support_embeddings ,support_labels_tensor ,query_embedding
            )
            loss =F .huber_loss (prediction .squeeze (),query_label_tensor .squeeze ())
        if not torch .isnan (loss ):
            total_loss_for_batch =total_loss_for_batch +loss
            queries_processed +=1
    if queries_processed >0 :
        loss =total_loss_for_batch /queries_processed
    else :
        loss =None
    return loss ,progress_bar_task
