import torch
import torch .nn as nn
import torch .nn .functional as F
from .meta_learner import MetaLearner
class MoEModel (nn .Module ):
    def __init__ (self ,num_experts ,strategy ,**kwargs ):
        super ().__init__ ()
        self .num_experts =num_experts
        self .strategy =strategy
        self .experts =nn .ModuleList ([
        MetaLearner (strategy =strategy ,**kwargs )
        for _ in range (num_experts )
        ])
        print (f"✅ Initialized MoEModel with {num_experts } expert(s).")
    def set_char_vocab (self ,char_to_id :dict ):
        if self .strategy =='learned':
            for expert in self .experts :
                expert .set_char_vocab (char_to_id )
    def _process_batch (self ,batch_of_sequences ):
        if not self .experts :
            return None
        all_expert_embeddings =[
        expert ._process_batch (batch_of_sequences )
        for expert in self .experts
        ]
        stacked_embeddings =torch .stack (all_expert_embeddings )
        avg_embeddings =torch .mean (stacked_embeddings ,dim =0 )
        return avg_embeddings
    def _aggregate_outputs (self ,expert_outputs ,task_type ,true_labels ):
        if task_type =='regression':
            all_preds =torch .stack ([out [0 ]for out in expert_outputs ])
            all_confs =torch .stack ([out [2 ]for out in expert_outputs ])
            weighted_preds =all_preds *all_confs
            sum_weighted_preds =weighted_preds .sum (dim =0 )
            sum_confs =all_confs .sum (dim =0 ).clamp_min (1e-8 )
            final_preds =sum_weighted_preds /sum_confs
            final_confidence =all_confs .mean (dim =0 )
            return final_preds ,true_labels ,final_confidence
        elif task_type =='classification':
            all_confs_stacked =torch .stack ([out [2 ]for out in expert_outputs ])
            summed_confs =all_confs_stacked .sum (dim =0 )
            final_predictions =summed_confs
            norm_confs =F .normalize (summed_confs ,p =1 ,dim =-1 )
            final_confidence ,_ =torch .max (norm_confs ,dim =-1 )
            return final_predictions ,true_labels ,final_confidence
    def forward (self ,support_set ,query_set ,task_type ,expert_id =None ):
        if self .training :
            if expert_id is None :
                raise ValueError ("MoEModel.forward() requires an 'expert_id' during training.")
            if expert_id >=self .num_experts :
                raise IndexError (f"Invalid expert_id {expert_id }. Max is {self .num_experts -1 }.")
            return self .experts [expert_id ](support_set ,query_set ,task_type )
        else :
            expert_outputs =[]
            all_true_labels =None
            for expert in self .experts :
                preds ,labels ,confs =expert (support_set ,query_set ,task_type )
                if preds is None :continue
                expert_outputs .append ((preds ,labels ,confs ))
                if all_true_labels is None :
                    all_true_labels =labels
            if not expert_outputs :
                return None ,None ,None
            return self ._aggregate_outputs (expert_outputs ,task_type ,all_true_labels )
