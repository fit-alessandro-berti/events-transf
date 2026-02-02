import torch
import torch .nn as nn
import torch .nn .functional as F
def _l2_normalize (x :torch .Tensor ,eps :float =1e-8 )->torch .Tensor :
    return x /x .norm (p =2 ,dim =-1 ,keepdim =True ).clamp_min (eps )
class PrototypicalHead (nn .Module ):
    def __init__ (self ,init_logit_scale :float =5.0 ):
        super ().__init__ ()
        self .logit_scale =nn .Parameter (torch .tensor (float (init_logit_scale )))
        self ._proto_shrink =nn .Parameter (torch .tensor (-2.0 ))
    def forward_classification (self ,support_features ,support_labels ,query_features ):
        if support_features .numel ()==0 :
            return None ,None ,None
        support_features =_l2_normalize (support_features )
        query_features =_l2_normalize (query_features )
        unique_classes =torch .unique (support_labels )
        class_means =[]
        class_counts =[]
        for cls in unique_classes :
            idx =(support_labels ==cls )
            class_counts .append (idx .sum ())
            proto =support_features [idx ].mean (dim =0 )
            class_means .append (proto )
        class_means =torch .stack (class_means ,dim =0 )
        counts =torch .stack (class_counts ).float ().clamp_min (1 )
        global_centroid =support_features .mean (dim =0 ,keepdim =True )
        alpha_base =torch .sigmoid (self ._proto_shrink ).clamp (0.0 ,0.4 )
        alpha_per_class =(alpha_base /counts .sqrt ()).unsqueeze (1 )
        prototypes =(1.0 -alpha_per_class )*class_means +alpha_per_class *global_centroid
        prototypes =_l2_normalize (prototypes )
        scale =self .logit_scale .clamp (1.0 ,100.0 )
        logits =(query_features @prototypes .t ())*scale
        confidence =F .softmax (logits ,dim =-1 )
        return logits ,unique_classes ,confidence
    def forward_regression (self ,support_features ,support_labels ,query_features ,eps :float =1e-6 ):
        if support_features .numel ()==0 or query_features .numel ()==0 :
            device =query_features .device
            return torch .zeros (query_features .size (0 ),device =device ),torch .zeros (query_features .size (0 ),device =device )
        support_features_norm =_l2_normalize (support_features )
        query_features_norm =_l2_normalize (query_features )
        distances_sq =torch .cdist (query_features_norm ,support_features_norm ).pow (2 )
        with torch .no_grad ():
            median_dist =torch .median (distances_sq .detach ())
        if not torch .isfinite (median_dist )or median_dist <=0 :
            median_dist =distances_sq .mean ()
        gamma =1.0 /(median_dist +eps )
        weights =F .softmax (-gamma *distances_sq ,dim =1 )
        prediction =weights @support_labels .view (-1 )
        confidence =torch .max (weights ,dim =1 ).values
        return prediction ,confidence
