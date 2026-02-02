import torch
import numpy as np
def find_knn_indices (query_embedding_norm :torch .Tensor ,
search_embeddings_norm :torch .Tensor ,
k :int ,
indices_to_mask :torch .Tensor =None ):
    if k <=0 :
        return torch .tensor ([],dtype =torch .long ,device =query_embedding_norm .device )
    sims =query_embedding_norm @search_embeddings_norm .T
    if indices_to_mask is not None and indices_to_mask .numel ()>0 :
        sims [0 ,indices_to_mask ]=-float ('inf')
    num_valid =(sims [0 ]>-float ('inf')).sum ().item ()
    k_to_find =min (k ,num_valid )
    if k_to_find <=0 :
        return torch .tensor ([],dtype =torch .long ,device =sims .device )
    top_k_indices =torch .topk (sims .squeeze (0 ),k_to_find ).indices
    return top_k_indices
