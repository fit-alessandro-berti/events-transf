import torch
import torch .nn .functional as F
import random
import numpy as np
from sklearn .metrics import accuracy_score ,mean_absolute_error ,r2_score
from sklearn .model_selection import GroupShuffleSplit ,train_test_split
from sklearn .ensemble import RandomForestClassifier ,RandomForestRegressor
from sklearn .ensemble import HistGradientBoostingRegressor
from sklearn .kernel_approximation import Nystroem
from sklearn .linear_model import HuberRegressor ,LogisticRegression ,Ridge ,SGDClassifier
from sklearn .pipeline import Pipeline
from sklearn .preprocessing import StandardScaler
from sklearn .svm import LinearSVC
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
def _build_classifiers ():
    return [
    (
    "RandomForest",
    RandomForestClassifier (
    n_estimators =800 ,
    random_state =42 ,
    n_jobs =-1 ,
    class_weight ="balanced_subsample",
    max_depth =8 ,
    min_samples_leaf =5 ,
    min_samples_split =10 ,
    max_features ="sqrt"
    ),
    ),
    (
    "StandardScaler+LinearSVC",
    Pipeline ([
    ("scaler",StandardScaler ()),
    ("model",LinearSVC (
    C =0.1 ,
    class_weight ="balanced",
    tol =1e-4 ,
    max_iter =20000 ,
    dual =True
    )),
    ]),
    ),
    ]
def _build_regressors ():
    return [
    (
    "RandomForest",
    RandomForestRegressor (
    n_estimators =800 ,
    random_state =42 ,
    n_jobs =-1 ,
    max_depth =10 ,
    min_samples_leaf =6 ,
    min_samples_split =10 ,
    max_features =0.5
    ),
    ),
    (
    "StandardScaler+Ridge",
    Pipeline ([
    ("scaler",StandardScaler ()),
    ("model",Ridge (alpha =50.0 )),
    ]),
    ),
    (
    "HistGradientBoostingRegressor",
    HistGradientBoostingRegressor (
    random_state =42 ,
    max_depth =3 ,
    learning_rate =0.05 ,
    max_iter =400 ,
    min_samples_leaf =10 ,
    l2_regularization =2.0 ,
    early_stopping =True ,
    validation_fraction =0.2 ,
    n_iter_no_change =20
    ),
    ),
    ]
def _compute_sample_metrics (labels_hours ,preds_hours ,case_test ):
    abs_errors =np .abs (labels_hours -preds_hours )
    sq_errors =(labels_hours -preds_hours )**2
    mae =float (np .mean (abs_errors ))if abs_errors .size else float ("nan")
    rmse =float (np .sqrt (np .mean (sq_errors )))if sq_errors .size else float ("nan")
    num_cases =len (np .unique (case_test ))if case_test is not None else 0
    return mae ,rmse ,num_cases
def _subsample_training_set (x_train ,y_train ,train_percentage ,stratify =None ):
    if train_percentage is None :
        return x_train ,y_train
    try :
        train_percentage =float (train_percentage )
    except (TypeError ,ValueError ):
        return x_train ,y_train
    if train_percentage >=100 or len (x_train )<2 :
        return x_train ,y_train
    train_percentage =max (1.0 ,min (train_percentage ,100.0 ))
    sample_size =max (1 ,int (np .ceil (len (x_train )*(train_percentage /100.0 ))))
    if sample_size >=len (x_train ):
        return x_train ,y_train
    try :
        x_sub ,_ ,y_sub ,_ =train_test_split (
        x_train ,
        y_train ,
        train_size =sample_size ,
        random_state =42 ,
        stratify =stratify
        )
        return x_sub ,y_sub
    except ValueError :
        rng =np .random .default_rng (42 )
        idx =rng .choice (len (x_train ),size =sample_size ,replace =False )
        return x_train [idx ],y_train [idx ]
def _report_sklearn_metrics (
expert_name :str ,
task_type :str ,
embeddings :torch .Tensor ,
labels :torch .Tensor ,
case_ids :np .ndarray =None ,
train_percentage =100
):
    x =embeddings .detach ().cpu ().numpy ()
    y =labels .detach ().cpu ().numpy ()
    num_samples =x .shape [0 ]
    if num_samples <2 :
        print (f"  - [{expert_name }] sklearn metrics skipped ({task_type }): not enough samples.")
        return
    if task_type =='classification':
        unique_labels ,counts =np .unique (y ,return_counts =True )
        if unique_labels .size <2 :
            print (f"  - [{expert_name }] sklearn metrics skipped (classification): only one class.")
            return
        stratify =y if counts .min ()>=2 else None
        if case_ids is not None :
            try :
                splitter =GroupShuffleSplit (
                n_splits =1 ,
                test_size =0.2 ,
                random_state =42
                )
                train_idx ,test_idx =next (splitter .split (x ,y ,groups =case_ids ))
                x_train =x [train_idx ]
                x_test =x [test_idx ]
                y_train =y [train_idx ]
                y_test =y [test_idx ]
            except ValueError as e :
                print (
                f"  - [{expert_name }] sklearn metrics: group split failed; "
                f"falling back to random split. Reason: {e }"
                )
                try :
                    x_train ,x_test ,y_train ,y_test =train_test_split (
                    x ,y ,test_size =0.2 ,random_state =42 ,stratify =stratify
                    )
                except ValueError :
                    x_train ,x_test ,y_train ,y_test =train_test_split (
                    x ,y ,test_size =0.2 ,random_state =42 ,stratify =None
                    )
        else :
            try :
                x_train ,x_test ,y_train ,y_test =train_test_split (
                x ,y ,test_size =0.2 ,random_state =42 ,stratify =stratify
                )
            except ValueError :
                x_train ,x_test ,y_train ,y_test =train_test_split (
                x ,y ,test_size =0.2 ,random_state =42 ,stratify =None
                )
        x_train ,y_train =_subsample_training_set (x_train ,y_train ,train_percentage ,stratify )
        if len (x_train )<2 :
            print (f"  - [{expert_name }] sklearn metrics skipped (classification): not enough training samples.")
            return
        if np .unique (y_train ).size <2 :
            print (f"  - [{expert_name }] sklearn metrics skipped (classification): training set has one class.")
            return
        for model_name ,clf in _build_classifiers ():
            try :
                clf .fit (x_train ,y_train )
                preds =clf .predict (x_test )
                acc =accuracy_score (y_test ,preds )
                print (
                f"  - [{expert_name }] {model_name } (classification, 80/20, train={train_percentage }%): "
                f"Accuracy={acc :.4f} (n={len (y_test )})"
                )
            except Exception as e :
                print (
                f"  - [{expert_name }] {model_name } (classification) failed: {e }"
                )
    else :
        if case_ids is None :
            case_ids =np .arange (len (y ))
        try :
            splitter =GroupShuffleSplit (
            n_splits =1 ,
            test_size =0.2 ,
            random_state =42
            )
            train_idx ,test_idx =next (splitter .split (x ,y ,groups =case_ids ))
            x_train =x [train_idx ]
            x_test =x [test_idx ]
            y_train =y [train_idx ]
            y_test =y [test_idx ]
            case_test =case_ids [test_idx ]
        except ValueError as e :
            print (
            f"  - [{expert_name }] sklearn metrics: group split failed; "
            f"falling back to random split. Reason: {e }"
            )
            x_train ,x_test ,y_train ,y_test ,_ ,case_test =train_test_split (
            x ,y ,case_ids ,test_size =0.2 ,random_state =42
            )
        x_train ,y_train =_subsample_training_set (x_train ,y_train ,train_percentage )
        if len (x_train )<2 :
            print (f"  - [{expert_name }] sklearn metrics skipped (regression): not enough training samples.")
            return
        for model_name ,reg in _build_regressors ():
            try :
                reg .fit (x_train ,y_train )
                preds =reg .predict (x_test )
                preds =np .asarray (preds ).reshape (-1 )
                preds_hours =inverse_transform_time (preds )
                preds_hours [preds_hours <0 ]=0
                labels_hours =inverse_transform_time (np .array (y_test ))
                mae ,rmse ,num_cases =_compute_sample_metrics (labels_hours ,preds_hours ,case_test )
                if len (labels_hours )<2 :
                    r2 =float ("nan")
                else :
                    r2 =r2_score (labels_hours ,preds_hours )
                print (
                f"  - [{expert_name }] {model_name } (regression, 80/20, train={train_percentage }%): "
                f"MAE(sample)={mae :.4f} | RMSE(sample)={rmse :.4f} | R2={r2 :.4f} "
                f"(cases={num_cases } | samples={len (y_test )})"
                )
            except Exception as e :
                print (
                f"  - [{expert_name }] {model_name } (regression) failed: {e }"
                )
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
def evaluate_retrieval_augmented (
model ,
test_tasks ,
num_retrieval_k_list ,
num_test_queries =200 ,
candidate_percentages =None ,
sklearn_train_percentage =100 ,
first_expert_only =False
):
    print ("\n🔬 Starting Retrieval-Augmented Evaluation...")
    model .eval ()
    if not candidate_percentages :
        candidate_percentages =[100 ]
    if hasattr (model ,"experts"):
        num_experts =len (model .experts )
    else :
        num_experts =1
    if num_experts >1 :
        experts_to_eval =[(f"Expert {i }",model .experts [i ])for i in range (num_experts )]
        print (f"  - (MoE) Running k-NN eval for all {num_experts } experts.")
    else :
        experts_to_eval =[("Expert 0",model )]
    if first_expert_only and len (experts_to_eval )>1 :
        experts_to_eval =experts_to_eval [:1 ]
        print ("  - Retrieval-augmented: limiting to first expert only.")
    expert_task_embeddings ={}
    for expert_name ,expert in experts_to_eval :
        expert_task_embeddings [expert_name ]={}
        for task_type ,task_data in test_tasks .items ():
            if not task_data :
                print (f"Skipping {task_type }: No test data available.")
                continue
            embeddings_raw ,labels ,case_ids =_get_all_test_embeddings (expert ,task_data )
            if embeddings_raw is None :
                return
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
            embeddings_norm =F .normalize (embeddings_raw ,p =2 ,dim =1 )
            if task_type =='classification':
                _report_similarity_metrics (embeddings_norm ,labels ,label =expert_name )
            expert_task_embeddings [expert_name ][task_type ]=(
            embeddings_norm ,
            labels ,
            case_ids ,
            embeddings_raw
            )
            print (f"  - [{expert_name }] Pre-computed {embeddings_norm .shape [0 ]} embeddings for {task_type }.")
    for task_type in test_tasks .keys ():
        _report_inter_expert_metrics (expert_task_embeddings ,task_type )
    for task_type in test_tasks .keys ():
        available_experts =[]
        for expert_name ,expert in experts_to_eval :
            if task_type in expert_task_embeddings .get (expert_name ,{}):
                embeddings_norm ,labels ,case_ids ,embeddings_raw =expert_task_embeddings [expert_name ][task_type ]
                available_experts .append ((
                expert_name ,
                expert ,
                embeddings_norm ,
                labels ,
                case_ids ,
                embeddings_raw
                ))
        if not available_experts :
            continue
        base_num_samples =available_experts [0 ][2 ].shape [0 ]
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
        for expert_name ,expert ,all_embeddings_norm ,all_labels ,all_case_ids ,all_embeddings_raw in available_experts :
            print (f"\n--- Evaluating task: {task_type } | {expert_name } ---")
            num_total_samples =all_embeddings_norm .shape [0 ]
            if num_total_samples <2 :
                print (f"Skipping {task_type } | {expert_name }: Not enough samples to evaluate.")
                continue
            if num_total_samples !=base_num_samples :
                print (
                f"  - Warning: {expert_name } has {num_total_samples } samples; "
                f"expected {base_num_samples }. Re-sampling queries for this expert."
                )
                num_queries =min (num_test_queries ,num_total_samples )
                query_indices =random .sample (range (num_total_samples ),num_queries )
                expert_candidate_pool_masks ={}
                for pct in candidate_percentages :
                    if pct >=100 :
                        expert_candidate_pool_masks [pct ]=None
                    elif pct <=0 :
                        expert_candidate_pool_masks [pct ]=np .arange (num_total_samples )
                    else :
                        sample_size =int (np .ceil (num_total_samples *(pct /100.0 )))
                        sample_size =max (1 ,min (sample_size ,num_total_samples ))
                        if sample_size ==num_total_samples :
                            expert_candidate_pool_masks [pct ]=None
                        else :
                            candidate_pool_indices =np .random .choice (
                            num_total_samples ,
                            size =sample_size ,
                            replace =False
                            )
                            mask_np =np .ones (num_total_samples ,dtype =bool )
                            mask_np [candidate_pool_indices ]=False
                            expert_candidate_pool_masks [pct ]=np .where (mask_np )[0 ]
            else :
                query_indices =base_query_indices
                expert_candidate_pool_masks =candidate_pool_masks
            for pct in candidate_percentages :
                print (f"\n  - Candidate pool sampling: {pct }%")
                non_candidate_indices_np =expert_candidate_pool_masks .get (pct )
                if non_candidate_indices_np is None :
                    non_candidate_indices_tensor =None
                else :
                    non_candidate_indices_tensor =torch .from_numpy (non_candidate_indices_np ).to (
                    all_embeddings_norm .device
                    )
                if non_candidate_indices_tensor is not None :
                    print (f"  - Candidate pool size: {num_total_samples -len (non_candidate_indices_np )} / {num_total_samples }")
                for k in num_retrieval_k_list :
                    if k >=num_total_samples :
                        print (f"Skipping [{expert_name } | k={k } | pct={pct }%]: k is larger than total samples.")
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
                        with torch .no_grad ():
                            if task_type =='classification':
                                logits ,proto_classes ,confidence =expert .proto_head .forward_classification (
                                support_embeddings ,support_labels ,query_embedding
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
                                prediction ,confidence =expert .proto_head .forward_regression (
                                support_embeddings ,support_labels .float (),query_embedding
                                )
                                all_preds .append (prediction [0 ].item ())
                                all_true_labels .append (query_label .item ())
                                all_confidences .append (confidence [0 ].item ())
                    if not all_true_labels :
                        print (f"Skipping [{expert_name } | k={k } | pct={pct }%]: no valid queries.")
                        continue
                    if task_type =='classification':
                        avg_conf =np .mean (all_confidences )
                        print (
                        f"[{expert_name } | {k }-NN | pct={pct }%] "
                        f"Retrieval Accuracy: {accuracy_score (all_true_labels ,all_preds ):.4f} | "
                        f"Avg. Confidence: {avg_conf :.4f} (on {len (all_true_labels )} queries)"
                        )
                    else :
                        preds_np =np .array (all_preds )
                        labels_np =np .array (all_true_labels )
                        avg_conf =np .mean (all_confidences )
                        preds =inverse_transform_time (preds_np )
                        preds [preds <0 ]=0
                        labels =inverse_transform_time (labels_np )
                        print (
                        f"[{expert_name } | {k }-NN | pct={pct }%] "
                        f"Retrieval MAE: {mean_absolute_error (labels ,preds ):.4f} | "
                        f"R-squared: {r2_score (labels ,preds ):.4f} | "
                        f"Avg. Confidence: {avg_conf :.4f}"
                        )
                _report_sklearn_metrics (
                f"{expert_name } | pct={pct }%",
                task_type ,
                all_embeddings_raw ,
                all_labels ,
                all_case_ids ,
                train_percentage =pct
                )
