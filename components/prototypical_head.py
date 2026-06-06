import torch
import torch .nn as nn
import torch .nn .functional as F
def _l2_normalize (x :torch .Tensor ,eps :float =1e-8 )->torch .Tensor :
    return x /x .norm (p =2 ,dim =-1 ,keepdim =True ).clamp_min (eps )
class PrototypicalHead (nn .Module ):
    def __init__ (self ,init_logit_scale :float =5.0 ):
        super ().__init__ ()
        self .logit_scale =nn .Parameter (torch .tensor (float (init_logit_scale )))
        self .reg_logit_scale =nn .Parameter (torch .tensor (float (init_logit_scale )))
        self ._proto_shrink =nn .Parameter (torch .tensor (-2.0 ))
        self .count_prior =nn .Parameter (torch .tensor (0.0 ))
    def _center_and_renorm (self ,support_features :torch .Tensor ,query_features :torch .Tensor ):
        mu =support_features .mean (dim =0 ,keepdim =True )
        support_centered =_l2_normalize (support_features -mu )
        query_centered =_l2_normalize (query_features -mu )
        return support_centered ,query_centered
    def forward_classification (self ,support_features ,support_labels ,query_features ,mode :str ="proto"):
        if support_features .numel ()==0 :
            return None ,None ,None
        support =_l2_normalize (support_features )
        query =_l2_normalize (query_features )
        unique_classes ,inv =torch .unique (support_labels ,sorted =True ,return_inverse =True )
        if mode =="soft_knn":
            support_centered ,query_centered =self ._center_and_renorm (support ,query )
            scale =self .logit_scale .clamp (1.0 ,20.0 )
            sims =(query_centered @support_centered .t ()) *scale
            attn =F .softmax (sims ,dim =1 )
            num_queries =attn .size (0 )
            num_classes =unique_classes .size (0 )
            class_mass =torch .zeros (num_queries ,num_classes ,device =attn .device )
            class_mass .scatter_add_ (1 ,inv .unsqueeze (0 ).expand (num_queries ,-1 ),attn )
            logits =torch .log (class_mass .clamp_min (1e-8 ))
            counts =torch .bincount (inv ,minlength =num_classes ).float ().clamp_min (1.0 )
            logits =logits +self .count_prior *torch .log (counts ).unsqueeze (0 )
            confidence =F .softmax (logits ,dim =-1 )
            return logits ,unique_classes ,confidence
        if mode =="proto":
            class_means =[]
            class_counts =[]
            for cls in unique_classes :
                idx =(support_labels ==cls )
                class_counts .append (idx .sum ())
                class_means .append (support [idx ].mean (dim =0 ))
            class_means =torch .stack (class_means ,dim =0 )
            counts =torch .stack (class_counts ).float ().clamp_min (1.0 )
            global_centroid =support .mean (dim =0 ,keepdim =True )
            alpha_base =torch .sigmoid (self ._proto_shrink ).clamp (0.0 ,0.4 )
            alpha_per_class =(alpha_base /counts .sqrt ()).unsqueeze (1 )
            prototypes =(1.0 -alpha_per_class )*class_means +alpha_per_class *global_centroid
            prototypes =_l2_normalize (prototypes )
            scale =self .logit_scale .clamp (1.0 ,100.0 )
            logits =(query @prototypes .t ()) *scale
            confidence =F .softmax (logits ,dim =-1 )
            return logits ,unique_classes ,confidence
        raise ValueError (f"Unknown mode: {mode }")
    def forward_regression (self ,support_features ,support_labels ,query_features ,eps :float =1e-6 ):
        if support_features .numel ()==0 or query_features .numel ()==0 :
            device =query_features .device
            return torch .zeros (query_features .size (0 ),device =device ),torch .zeros (query_features .size (0 ),device =device )
        support =_l2_normalize (support_features )
        query =_l2_normalize (query_features )
        distances_sq =torch .cdist (query ,support ).pow (2 )
        with torch .no_grad ():
            median_dist =torch .median (distances_sq .detach ())
        if not torch .isfinite (median_dist )or median_dist <=0 :
            median_dist =distances_sq .mean ()
        gamma =1.0 /(median_dist +eps )
        weights =F .softmax (-gamma *distances_sq ,dim =1 )
        prediction =weights @support_labels .view (-1 ).float ()
        confidence =torch .max (weights ,dim =1 ).values
        return prediction ,confidence
