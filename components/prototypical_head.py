import torch
import torch .nn as nn
import torch .nn .functional as F
def _l2_normalize (x :torch .Tensor ,eps :float =1e-8 )->torch .Tensor :
    return x /x .norm (p =2 ,dim =-1 ,keepdim =True ).clamp_min (eps )
class PrototypicalHead (nn .Module ):
    def __init__ (self ,init_logit_scale :float =5.0 ):
        super ().__init__ ()
        self .logit_scale =nn .Parameter (torch .tensor (float (init_logit_scale )))
        self .logit_scale .requires_grad_(False )
        self .reg_logit_scale =nn .Parameter (torch .tensor (float (init_logit_scale )))
        self ._proto_shrink =nn .Parameter (torch .tensor (-2.0 ))
        self .count_prior =nn .Parameter (torch .tensor (0.0 ))
    def _center_and_renorm (self ,support_features :torch .Tensor ,query_features :torch .Tensor ):
        mu =support_features .mean (dim =0 ,keepdim =True )
        support_centered =_l2_normalize (support_features -mu )
        query_centered =_l2_normalize (query_features -mu )
        return support_centered ,query_centered
    def forward_classification (self ,support_features ,support_labels ,query_features ,mode :str ="soft_knn"):
        if support_features .numel ()==0 :
            return None ,None ,None
        support =_l2_normalize (support_features )
        query =_l2_normalize (query_features )
        support ,query =self ._center_and_renorm (support ,query )
        unique_classes ,inv =torch .unique (support_labels ,sorted =True ,return_inverse =True )
        scale =self .logit_scale .clamp (1.0 ,20.0 )
        if mode =="soft_knn":
            sims =(query @support .t ()) *scale
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
            alpha_per_class =torch .where (
            counts .unsqueeze (1 )<3.0 ,
            torch .zeros_like (alpha_per_class ),
            alpha_per_class
            )
            prototypes =(1.0 -alpha_per_class )*class_means +alpha_per_class *global_centroid
            prototypes =_l2_normalize (prototypes )
            logits =(query @prototypes .t ()) *scale
            confidence =F .softmax (logits ,dim =-1 )
            return logits ,unique_classes ,confidence
        raise ValueError (f"Unknown mode: {mode }")
    def forward_regression (self ,support_features ,support_labels ,query_features ):
        if support_features .numel ()==0 or query_features .numel ()==0 :
            device =query_features .device
            return torch .zeros (query_features .size (0 ),device =device ),torch .zeros (query_features .size (0 ),device =device )
        support =_l2_normalize (support_features )
        query =_l2_normalize (query_features )
        support ,query =self ._center_and_renorm (support ,query )
        scale =self .reg_logit_scale .clamp (1.0 ,100.0 )
        sims_raw =query @support .t ()
        sims =sims_raw *scale
        weights =F .softmax (sims ,dim =1 )
        prediction =weights @support_labels .view (-1 ).float ()
        support_targets =support_labels .view (1 ,-1 ).float ()
        pred_center =prediction .view (-1 ,1 )
        var =(weights *(support_targets -pred_center )**2 ).sum (dim =1 )
        std =torch .sqrt (var +1e-8 )
        conf_consensus =1.0 /(1.0 +std )
        conf_similarity =((sims_raw .max (dim =1 ).values +1.0 )/2.0 ).clamp (0.0 ,1.0 )
        confidence =(conf_consensus *conf_similarity ).clamp (0.0 ,1.0 )
        return prediction ,confidence
