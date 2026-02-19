import torch
import torch .nn .functional as F
import random
import numpy as np
from sklearn .metrics import accuracy_score ,mean_absolute_error ,r2_score
from tqdm import tqdm
from time_transf import inverse_transform_time
from utils .retrieval_utils import find_knn_indices
def _report_similarity_metrics (
embeddings :torch .Tensor ,
labels :torch .Tensor ,
max_knn_queries =1000 ,
knn_k_list =(5 ,10 ),
label :str =None
):
    if embeddings .numel ()==0 or labels .numel ()==0 :
        print ("  - Similarity metrics: skipped (empty embeddings/labels).")
        return
    device =embeddings .device
    labels =labels .to (device )
    mean_emb =embeddings .mean (dim =0 ,keepdim =True )
    centered =embeddings -mean_emb
    centered =F .normalize (centered ,p =2 ,dim =1 )
    unique_labels ,inverse =torch .unique (labels ,sorted =True ,return_inverse =True )
    num_classes =unique_labels .numel ()
    if num_classes <2 :
        print (f"  - Similarity metrics: skipped (num_classes={num_classes }).")
        return
    n ,d =centered .shape
    centroids =torch .zeros ((num_classes ,d ),device =device )
    counts =torch .zeros ((num_classes ,),device =device )
    centroids .scatter_add_ (0 ,inverse [:,None ].expand (-1 ,d ),centered )
    counts .scatter_add_ (0 ,inverse ,torch .ones ((n ,),device =device ))
    centroids =centroids /counts [:,None ].clamp_min (1.0 )
    centroids =F .normalize (centroids ,p =2 ,dim =1 )
    sims_to_own =(centered *centroids [inverse ]).sum (dim =1 )
    intra_mean =sims_to_own .mean ().item ()
    centroid_sims =centroids @centroids .T
    triu_idx =torch .triu_indices (num_classes ,num_classes ,offset =1 ,device =device )
    inter_mean =centroid_sims [triu_idx [0 ],triu_idx [1 ]].mean ().item ()
    centroid_sims_offdiag =centroids @centroids .T
    centroid_sims_offdiag .fill_diagonal_ (-float ('inf'))
    max_other_centroid_sim =centroid_sims_offdiag .max (dim =1 ).values
    margin_mean =(1.0 -max_other_centroid_sim ).mean ().item ()
    knn_purity ={}
    max_queries =min (max_knn_queries ,n )
    if max_queries <n :
        query_idx =torch .randperm (n ,device =device )[:max_queries ]
    else :
        query_idx =torch .arange (n ,device =device )
    query_emb =centered [query_idx ]
    query_labels =labels [query_idx ]
    sims =query_emb @centered .T
    sims [torch .arange (max_queries ,device =device ),query_idx ]=-float ('inf')
    for k in knn_k_list :
        k_eff =min (k ,n -1 )
        if k_eff <=0 :
            knn_purity [k ]=float ('nan')
            continue
        topk_idx =torch .topk (sims ,k_eff ,dim =1 ).indices
        neighbor_labels =labels [topk_idx ]
        purity =(neighbor_labels ==query_labels [:,None ]).float ().mean ().item ()
        knn_purity [k ]=purity
    label_prefix =f"[{label }] "if label else ""
    print (
    "  - "
    +label_prefix
    +"Similarity metrics (mean, mean-centered cosine): "
    f"intra_centroid_cos={intra_mean :.4f} | "
    f"inter_centroid_cos={inter_mean :.4f} | "
    f"centroid_margin={margin_mean :.4f} | "
    +" ".join ([f"knn_purity@{k }={knn_purity [k ]:.4f}"for k in knn_k_list ])
    )
def _report_inter_expert_metrics (expert_task_embeddings ,task_type :str ):
    expert_names =list (expert_task_embeddings .keys ())
    if len (expert_names )<2 :
        return
    pairs =[]
    for i in range (len (expert_names )):
        for j in range (i +1 ,len (expert_names )):
            pairs .append ((expert_names [i ],expert_names [j ]))
    for expert_a ,expert_b in pairs :
        data_a =expert_task_embeddings [expert_a ].get (task_type )
        data_b =expert_task_embeddings [expert_b ].get (task_type )
        if data_a is None or data_b is None :
            continue
        emb_a ,labels_a ,_ ,_ =data_a
        emb_b ,labels_b ,_ ,_ =data_b
        if emb_a .shape [0 ]!=emb_b .shape [0 ]:
            print (
            f"  - Inter-expert metrics skipped for {expert_a } vs {expert_b } "
            f"({task_type }): sample count mismatch."
            )
            continue
        mean_cos =(emb_a *emb_b ).sum (dim =1 ).mean ().item ()
        mean_l2 =torch .norm (emb_a -emb_b ,dim =1 ).mean ().item ()
        if task_type =='classification':
            unique_labels =torch .unique (labels_a ,sorted =True )
            if unique_labels .numel ()<2 :
                centroid_cos_mean =float ('nan')
            else :
                centroids_a =[]
                centroids_b =[]
                for lbl in unique_labels :
                    mask_a =labels_a ==lbl
                    mask_b =labels_b ==lbl
                    if not mask_a .any ()or not mask_b .any ():
                        continue
                    ca =emb_a [mask_a ].mean (dim =0 ,keepdim =True )
                    cb =emb_b [mask_b ].mean (dim =0 ,keepdim =True )
                    ca =F .normalize (ca ,p =2 ,dim =1 )
                    cb =F .normalize (cb ,p =2 ,dim =1 )
                    centroids_a .append (ca )
                    centroids_b .append (cb )
                if not centroids_a :
                    centroid_cos_mean =float ('nan')
                else :
                    centroids_a =torch .cat (centroids_a ,dim =0 )
                    centroids_b =torch .cat (centroids_b ,dim =0 )
                    centroid_cos_mean =(centroids_a *centroids_b ).sum (dim =1 ).mean ().item ()
            print (
            f"  - Inter-expert metrics ({task_type }) {expert_a } vs {expert_b }: "
            f"mean_cos={mean_cos :.4f} | mean_l2={mean_l2 :.4f} | "
            f"centroid_cos_mean={centroid_cos_mean :.4f}"
            )
        else :
            print (
            f"  - Inter-expert metrics ({task_type }) {expert_a } vs {expert_b }: "
            f"mean_cos={mean_cos :.4f} | mean_l2={mean_l2 :.4f}"
            )
def _to_hours (values ):
    hours =inverse_transform_time (np .asarray (values ,dtype =float ))
    return np .maximum (hours ,0.0 )
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
        print ("❌ ERROR in _get_all_test_embeddings:")
        print ("Test data does not contain case_ids.")
        print ("Please modify get_task_data in data_generator.py to return:")
        print ("(prefix, label, case_id) tuples.")
        print ("Aborting retrieval-augmented evaluation.")
        print ("="*50 +"\n")
        return None ,None ,None
    with torch .no_grad ():
        for i in tqdm (range (0 ,len (test_tasks_list ),batch_size ),desc ="Pre-computing test embeddings"):
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
def _predict_feature_knn_classification (
support_labels :torch .Tensor ,
support_sims :torch .Tensor
):
    if support_labels is None or support_labels .numel ()==0 :
        return None ,None
    support_labels =support_labels .view (-1 ).long ()
    support_sims =support_sims .view (-1 ).float ()
    unique_labels ,inverse =torch .unique (support_labels ,sorted =True ,return_inverse =True )
    class_counts =torch .bincount (inverse ,minlength =unique_labels .numel ()).float ()
    max_count =class_counts .max ()
    winners =(class_counts ==max_count ).nonzero (as_tuple =False ).view (-1 )
    if winners .numel ()==1 :
        winner_idx =winners [0 ]
    else :
        sim_sums =torch .zeros (unique_labels .numel (),device =support_sims .device ,dtype =support_sims .dtype )
        sim_sums .scatter_add_ (0 ,inverse ,support_sims )
        winner_local_idx =torch .argmax (sim_sums [winners ])
        winner_idx =winners [winner_local_idx ]
    pred_label =int (unique_labels [winner_idx ].item ())
    pred_conf =float ((class_counts [winner_idx ]/max (1.0 ,float (support_labels .numel ()))).item ())
    return pred_label ,pred_conf
def _predict_feature_knn_regression (
support_labels :torch .Tensor ,
support_sims :torch .Tensor
):
    if support_labels is None or support_labels .numel ()==0 :
        return None ,None
    preds =support_labels .view (-1 ).float ()
    pred_value =float (preds .mean ().item ())
    if support_sims is None or support_sims .numel ()==0 :
        pred_conf =0.0
    else :
        pred_conf =float (torch .clamp ((support_sims .float ().mean ()+1.0 )/2.0 ,0.0 ,1.0 ).item ())
    return pred_value ,pred_conf
def _report_confidence_bucket_metrics (
task_type :str ,
confidences ,
preds ,
true_labels ,
num_buckets :int =5
):
    if not confidences :
        print ("    Confidence buckets: skipped (no confidence values).")
        return
    conf =np .clip (np .asarray (confidences ,dtype =float ),0.0 ,1.0 )
    preds_np =np .asarray (preds )
    labels_np =np .asarray (true_labels )
    edges =np .linspace (0.0 ,1.0 ,num_buckets +1 )
    print ("    Confidence buckets:")
    for i in range (num_buckets ):
        low =edges [i ]
        high =edges [i +1 ]
        if i <num_buckets -1 :
            mask =(conf >=low )&(conf <high )
            bucket_label =f"[{low :.2f}, {high :.2f})"
        else :
            mask =(conf >=low )&(conf <=high )
            bucket_label =f"[{low :.2f}, {high :.2f}]"
        n =int (mask .sum ())
        if n ==0 :
            print (f"      - {bucket_label }: n=0")
            continue
        if task_type =='classification':
            acc =accuracy_score (labels_np [mask ],preds_np [mask ])
            print (f"      - {bucket_label }: n={n } | Accuracy={acc :.4f}")
        else :
            bucket_preds =_to_hours (preds_np [mask ])
            bucket_labels =_to_hours (labels_np [mask ])
            mae =mean_absolute_error (bucket_labels ,bucket_preds )
            if n <2 or np .unique (bucket_labels ).size <2 :
                r2_str ="nan"
            else :
                r2_val =r2_score (bucket_labels ,bucket_preds )
                r2_str =f"{r2_val :.4f}"
            print (f"      - {bucket_label }: n={n } | MAE={mae :.4f} | R-squared={r2_str }")
def evaluate_retrieval_augmented (
model ,
test_tasks ,
num_retrieval_k_list ,
num_test_queries =200 ,
candidate_percentages =None ,
first_expert_only =False ,
eval_scope ="experts",
prediction_mode ="proto_head",
report_confidence_buckets =False
):
    print ("\n🔬 Starting Retrieval-Augmented Evaluation...")
    model .eval ()
    if not candidate_percentages :
        candidate_percentages =[100 ]
    mode =str (prediction_mode or "proto_head").strip ().lower ()
    if mode in {"proto","proto_head","prototypical","prototypical_head"}:
        mode ="proto_head"
    elif mode in {"foundation_knn","feature_knn","knn"}:
        mode ="foundation_knn"
    else :
        print (f"  - Unknown prediction_mode '{prediction_mode }', falling back to 'proto_head'.")
        mode ="proto_head"
    print (f"  - Retrieval prediction mode: {mode }")
    if report_confidence_buckets and mode !="proto_head":
        print ("  - Confidence-bucket report is enabled, but ignored for non-proto_head modes.")
    scope =str (eval_scope or "experts").strip ().lower ()
    if scope not in {"experts","model"}:
        print (f"  - Unknown eval_scope '{eval_scope }', falling back to 'experts'.")
        scope ="experts"

    if hasattr (model ,"experts"):
        num_experts =len (model .experts )
        all_experts =[(f"Expert {i }",model .experts [i ])for i in range (num_experts )]
    else :
        num_experts =1
        all_experts =[("Model",model )]

    experts_for_eval =all_experts
    if first_expert_only and scope =="experts" and len (experts_for_eval )>1 :
        experts_for_eval =experts_for_eval [:1 ]
        print ("  - Retrieval-augmented: limiting to first expert only.")
    if first_expert_only and scope =="model" and len (all_experts )>1 :
        print ("  - Retrieval-augmented: eval_scope='model' ignores first_expert_only and uses all experts.")

    experts_for_embedding =all_experts if scope =="model" else experts_for_eval
    expert_task_embeddings ={}
    if num_experts >1 :
        print (f"  - (MoE) Retrieval evaluation scope: {scope }")
    for expert_name ,expert in experts_for_embedding :
        expert_task_embeddings [expert_name ]={}
        for task_type ,task_data in test_tasks .items ():
            if not task_data :
                continue
            embeddings_raw ,labels ,case_ids =_get_all_test_embeddings (expert ,task_data )
            if embeddings_raw is None :
                return
            embeddings_norm =F .normalize (embeddings_raw ,p =2 ,dim =1 )
            expert_task_embeddings [expert_name ][task_type ]=(
            embeddings_norm ,
            labels ,
            case_ids ,
            embeddings_raw
            )
            if scope =="experts":
                try :
                    has_nan =torch .isnan (embeddings_raw ).any ().item ()
                    all_finite =torch .isfinite (embeddings_raw ).all ().item ()
                    print (f"  - [{expert_name }] Embedding sanity: has_nan={has_nan }, all_finite={all_finite }")
                except Exception as e :
                    print (f"  - [{expert_name }] Embedding sanity check failed: {e }")
                if case_ids is not None :
                    unique_cases ,case_counts =np .unique (case_ids ,return_counts =True )
                    print (f"  - [{expert_name }] Case ID stats: unique_cases={len (unique_cases )}, total_tasks={len (case_ids )}")
                    if len (unique_cases )>0 :
                        top_k =min (5 ,len (unique_cases ))
                        top_idx =np .argsort (case_counts )[-top_k :][::-1 ]
                        top_cases =[(unique_cases [i ],int (case_counts [i ]))for i in top_idx ]
                        print (f"  - [{expert_name }] Top case_id counts: {top_cases }")
                if task_type =='classification':
                    _report_similarity_metrics (embeddings_norm ,labels ,label =expert_name )
                print (f"  - [{expert_name }] Pre-computed {embeddings_norm .shape [0 ]} embeddings for {task_type }.")

    model_task_embeddings ={}
    if scope =="model":
        for task_type ,task_data in test_tasks .items ():
            if not task_data :
                print (f"Skipping {task_type }: No test data available.")
                continue
            embeddings_raw ,labels ,case_ids =_get_all_test_embeddings (model ,task_data )
            if embeddings_raw is None :
                return
            embeddings_norm =F .normalize (embeddings_raw ,p =2 ,dim =1 )
            model_task_embeddings [task_type ]=(embeddings_norm ,labels ,case_ids ,embeddings_raw )
            try :
                has_nan =torch .isnan (embeddings_raw ).any ().item ()
                all_finite =torch .isfinite (embeddings_raw ).all ().item ()
                print (f"  - [Model] Embedding sanity: has_nan={has_nan }, all_finite={all_finite }")
            except Exception as e :
                print (f"  - [Model] Embedding sanity check failed: {e }")
            if case_ids is not None :
                unique_cases ,case_counts =np .unique (case_ids ,return_counts =True )
                print (f"  - [Model] Case ID stats: unique_cases={len (unique_cases )}, total_tasks={len (case_ids )}")
                if len (unique_cases )>0 :
                    top_k =min (5 ,len (unique_cases ))
                    top_idx =np .argsort (case_counts )[-top_k :][::-1 ]
                    top_cases =[(unique_cases [i ],int (case_counts [i ]))for i in top_idx ]
                    print (f"  - [Model] Top case_id counts: {top_cases }")
            if task_type =='classification':
                _report_similarity_metrics (embeddings_norm ,labels ,label ="Model" )
            print (f"  - [Model] Pre-computed {embeddings_norm .shape [0 ]} embeddings for {task_type }.")
    else :
        for task_type in test_tasks .keys ():
            _report_inter_expert_metrics (expert_task_embeddings ,task_type )

    for task_type in test_tasks .keys ():
        if scope =="model":
            if task_type not in model_task_embeddings :
                continue
            eval_units =[("Model",model ,*model_task_embeddings [task_type ])]
            experts_for_agg =[]
            for expert_name ,expert in all_experts :
                data =expert_task_embeddings .get (expert_name ,{}).get (task_type )
                if data is None :
                    continue
                experts_for_agg .append ((expert_name ,expert ,*data ))
        else :
            eval_units =[]
            experts_for_agg =[]
            for expert_name ,expert in experts_for_eval :
                data =expert_task_embeddings .get (expert_name ,{}).get (task_type )
                if data is None :
                    continue
                eval_units .append ((expert_name ,expert ,*data ))
            if not eval_units :
                continue

        if not eval_units :
            continue

        base_num_samples =eval_units [0 ][2 ].shape [0 ]
        if base_num_samples <2 :
            print (f"Skipping {task_type }: Not enough samples to evaluate.")
            continue
        num_queries =min (num_test_queries ,base_num_samples )
        base_query_indices =random .sample (range (base_num_samples ),num_queries )
        candidate_pool_masks ={}
        for pct in candidate_percentages :
            if pct >=100 :
                candidate_pool_masks [pct ]=None
            elif pct <=0 :
                candidate_pool_masks [pct ]=np .arange (base_num_samples )
            else :
                sample_size =int (np .ceil (base_num_samples *(pct /100.0 )))
                sample_size =max (1 ,min (sample_size ,base_num_samples ))
                if sample_size ==base_num_samples :
                    candidate_pool_masks [pct ]=None
                else :
                    candidate_pool_indices =np .random .choice (
                    base_num_samples ,
                    size =sample_size ,
                    replace =False
                    )
                    mask_np =np .ones (base_num_samples ,dtype =bool )
                    mask_np [candidate_pool_indices ]=False
                    candidate_pool_masks [pct ]=np .where (mask_np )[0 ]

        for unit_name ,unit_model ,all_embeddings_norm ,all_labels ,all_case_ids ,_ in eval_units :
            print (f"\n--- Evaluating task: {task_type } | {unit_name } ---")
            num_total_samples =all_embeddings_norm .shape [0 ]
            if num_total_samples <2 :
                print (f"Skipping {task_type } | {unit_name }: Not enough samples to evaluate.")
                continue
            if num_total_samples !=base_num_samples :
                print (
                f"  - Warning: {unit_name } has {num_total_samples } samples; "
                f"expected {base_num_samples }. Re-sampling queries for this unit."
                )
                num_queries =min (num_test_queries ,num_total_samples )
                query_indices =random .sample (range (num_total_samples ),num_queries )
                unit_candidate_pool_masks ={}
                for pct in candidate_percentages :
                    if pct >=100 :
                        unit_candidate_pool_masks [pct ]=None
                    elif pct <=0 :
                        unit_candidate_pool_masks [pct ]=np .arange (num_total_samples )
                    else :
                        sample_size =int (np .ceil (num_total_samples *(pct /100.0 )))
                        sample_size =max (1 ,min (sample_size ,num_total_samples ))
                        if sample_size ==num_total_samples :
                            unit_candidate_pool_masks [pct ]=None
                        else :
                            candidate_pool_indices =np .random .choice (
                            num_total_samples ,
                            size =sample_size ,
                            replace =False
                            )
                            mask_np =np .ones (num_total_samples ,dtype =bool )
                            mask_np [candidate_pool_indices ]=False
                            unit_candidate_pool_masks [pct ]=np .where (mask_np )[0 ]
            else :
                query_indices =base_query_indices
                unit_candidate_pool_masks =candidate_pool_masks

            for pct in candidate_percentages :
                print (f"\n  - Candidate pool sampling: {pct }%")
                non_candidate_indices_np =unit_candidate_pool_masks .get (pct )
                if non_candidate_indices_np is None :
                    non_candidate_indices_tensor =None
                else :
                    non_candidate_indices_tensor =torch .from_numpy (non_candidate_indices_np ).to (all_embeddings_norm .device )
                if non_candidate_indices_tensor is not None :
                    print (f"  - Candidate pool size: {num_total_samples -len (non_candidate_indices_np )} / {num_total_samples }")

                for k in num_retrieval_k_list :
                    if k >=num_total_samples :
                        print (f"Skipping [{unit_name } | k={k } | pct={pct }%]: k is larger than total samples.")
                        continue
                    all_preds ,all_true_labels ,all_confidences =[],[],[]
                    for query_idx in query_indices :
                        query_embedding =all_embeddings_norm [query_idx :query_idx +1 ]
                        query_label =all_labels [query_idx ]
                        query_case_id =all_case_ids [query_idx ]
                        same_case_indices_np =np .where (all_case_ids ==query_case_id )[0 ]
                        if non_candidate_indices_tensor is None :
                            if same_case_indices_np .size ==0 :
                                mask_tensor =None
                            else :
                                mask_tensor =torch .from_numpy (same_case_indices_np ).to (all_embeddings_norm .device )
                        else :
                            if same_case_indices_np .size ==0 :
                                mask_tensor =non_candidate_indices_tensor
                            else :
                                same_case_tensor =torch .from_numpy (same_case_indices_np ).to (all_embeddings_norm .device )
                                mask_tensor =torch .cat ([non_candidate_indices_tensor ,same_case_tensor ])
                        top_k_indices =find_knn_indices (
                        query_embedding ,
                        all_embeddings_norm ,
                        k =k ,
                        indices_to_mask =mask_tensor
                        )
                        if top_k_indices .numel ()==0 :
                            continue
                        support_embeddings =all_embeddings_norm [top_k_indices ]
                        support_labels =all_labels [top_k_indices ]
                        support_sims =(query_embedding @support_embeddings .T ).view (-1 )

                        if mode =="foundation_knn":
                            if task_type =='classification':
                                predicted_class_label ,pred_confidence =_predict_feature_knn_classification (
                                support_labels ,
                                support_sims
                                )
                                if predicted_class_label is None :
                                    continue
                                all_preds .append (predicted_class_label )
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (pred_confidence )
                            else :
                                pred_value ,pred_confidence =_predict_feature_knn_regression (
                                support_labels ,
                                support_sims
                                )
                                if pred_value is None :
                                    continue
                                all_preds .append (pred_value )
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (pred_confidence )
                            continue

                        if scope =="model" and len (experts_for_agg )>=1 :
                            expert_outputs =[]
                            proto_classes_ref =None
                            query_label_tensor =torch .as_tensor ([query_label .item ()],device =query_embedding .device )
                            for _expert_name ,expert_model ,expert_embeddings_norm ,expert_labels ,_expert_case_ids ,_expert_embeddings_raw in experts_for_agg :
                                if expert_embeddings_norm .shape [0 ]!=num_total_samples :
                                    continue
                                expert_query_embedding =expert_embeddings_norm [query_idx :query_idx +1 ]
                                expert_support_embeddings =expert_embeddings_norm [top_k_indices ]
                                expert_support_labels =expert_labels [top_k_indices ]
                                with torch .no_grad ():
                                    if task_type =='classification':
                                        logits ,proto_classes ,confidence =expert_model .proto_head .forward_classification (
                                        expert_support_embeddings ,expert_support_labels ,expert_query_embedding ,mode ="soft_knn"
                                        )
                                        if logits is None :
                                            continue
                                        if proto_classes_ref is None :
                                            proto_classes_ref =proto_classes
                                        expert_outputs .append ((logits ,query_label_tensor ,confidence ))
                                    else :
                                        prediction ,confidence =expert_model .proto_head .forward_regression (
                                        expert_support_embeddings ,expert_support_labels .float (),expert_query_embedding
                                        )
                                        expert_outputs .append ((prediction .view (-1 ),query_label_tensor .float (),confidence .view (-1 )))
                            if not expert_outputs :
                                continue
                            final_preds ,_ ,final_conf =model ._aggregate_outputs (expert_outputs ,task_type ,query_label_tensor )
                            if task_type =='classification':
                                pred_label_idx =torch .argmax (final_preds ,dim =1 ).item ()
                                pred_confidence =float (final_conf [0 ].item ()) if isinstance (final_conf ,torch .Tensor )else float (final_conf )
                                if proto_classes_ref is None :
                                    continue
                                predicted_class_label =proto_classes_ref [pred_label_idx ].item ()
                                all_preds .append (predicted_class_label )
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (pred_confidence )
                            else :
                                pred_value =float (final_preds [0 ].item ()) if isinstance (final_preds ,torch .Tensor )else float (final_preds )
                                pred_confidence =float (final_conf [0 ].item ()) if isinstance (final_conf ,torch .Tensor )else float (final_conf )
                                all_preds .append (pred_value )
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (pred_confidence )
                            continue

                        with torch .no_grad ():
                            if task_type =='classification':
                                logits ,proto_classes ,confidence =unit_model .proto_head .forward_classification (
                                support_embeddings ,support_labels ,query_embedding ,mode ="soft_knn"
                                )
                                if logits is None :
                                    continue
                                pred_label_idx =torch .argmax (logits ,dim =1 ).item ()
                                pred_confidence =confidence [0 ,pred_label_idx ].item ()
                                predicted_class_label =proto_classes [pred_label_idx ].item ()
                                all_preds .append (predicted_class_label )
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (pred_confidence )
                            else :
                                prediction ,confidence =unit_model .proto_head .forward_regression (
                                support_embeddings ,support_labels .float (),query_embedding
                                )
                                all_preds .append (prediction [0 ].item ())
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (confidence [0 ].item ())

                    if not all_true_labels :
                        print (f"Skipping [{unit_name } | k={k } | pct={pct }%]: no valid queries.")
                        continue
                    if task_type =='classification':
                        avg_conf =np .mean (all_confidences )
                        print (
                        f"[{unit_name } | {k }-NN | pct={pct }%] "
                        f"Retrieval Accuracy: {accuracy_score (all_true_labels ,all_preds ):.4f} | "
                        f"Avg. Confidence: {avg_conf :.4f} (on {len (all_true_labels )} queries)"
                        )
                    else :
                        preds_np =np .array (all_preds )
                        labels_np =np .array (all_true_labels )
                        avg_conf =np .mean (all_confidences )
                        preds =_to_hours (preds_np )
                        labels =_to_hours (labels_np )
                        print (
                        f"[{unit_name } | {k }-NN | pct={pct }%] "
                        f"Retrieval MAE: {mean_absolute_error (labels ,preds ):.4f} | "
                        f"R-squared: {r2_score (labels ,preds ):.4f} | "
                        f"Avg. Confidence: {avg_conf :.4f}"
                        )
                    if report_confidence_buckets and mode =="proto_head":
                        _report_confidence_bucket_metrics (
                        task_type ,
                        all_confidences ,
                        all_preds ,
                        all_true_labels ,
                        num_buckets =5
                        )
