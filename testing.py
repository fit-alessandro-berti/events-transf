import torch
import numpy as np
import os
import re
import warnings
import argparse
from pathlib import Path
from config import CONFIG
from time_transf import inverse_transform_time
from utils .data_utils import get_task_data
from utils .model_utils import init_loader ,create_model ,load_model_weights
from evaluation import evaluate_model ,evaluate_retrieval_augmented ,evaluate_pca_knn
if __name__ =='__main__':
    parser =argparse .ArgumentParser (description ="Run the meta-learning model evaluation script.")
    default_config =CONFIG
    parser .add_argument ('--checkpoint_dir',type =str ,default ='./checkpoints',help ="Directory to load checkpoints and artifacts from.")
    parser .add_argument ('--checkpoint_epoch',type =int ,default =None ,help ="Specific epoch checkpoint to test (e.g., 1, 5). Defaults to the latest.")
    parser .add_argument ('--test_log_name',type =str ,required =True ,help ="Name of the test log (from config) OR a direct path to a .xes.gz file.")
    parser .add_argument ('--test_mode',type =str ,default =default_config ['test_mode'],choices =['meta_learning','retrieval_augmented'],help =f"Evaluation mode. (default: {default_config ['test_mode']})")
    parser .add_argument ('--num_test_episodes',type =int ,default =default_config ['num_test_episodes'],help =f"Number of episodes to run for testing. (default: {default_config ['num_test_episodes']})")
    parser .add_argument ('--test_retrieval_k',type =int ,nargs ='+',default =default_config ['test_retrieval_k'],help =f"List of k-values for retrieval-augmented mode. (default: {default_config ['test_retrieval_k']})")
    parser .add_argument ('--test_retrieval_candidate_percentages',type =float ,nargs ='+',default =default_config .get ('test_retrieval_candidate_percentages',[100 ]),help ="List of candidate-pool sampling percentages for retrieval-augmented mode.")
    parser .add_argument (
    '--test_retrieval_eval_scope',
    type =str ,
    default =default_config .get ('test_retrieval_eval_scope','experts'),
    choices =['experts','model'],
    help ="Evaluate retrieval candidate pools per expert or for whole model."
    )
    parser .add_argument (
    '--test_retrieval_first_expert_only',
    action ='store_true',
    default =default_config .get ('test_retrieval_first_expert_only',False ),
    help ="If set, stop retrieval-augmented evaluation after the first expert."
    )
    args =parser .parse_args ()
    CONFIG ['test_mode']=args .test_mode
    CONFIG ['num_test_episodes']=args .num_test_episodes
    CONFIG ['test_retrieval_k']=args .test_retrieval_k
    CONFIG ['test_retrieval_candidate_percentages']=args .test_retrieval_candidate_percentages
    CONFIG ['test_retrieval_eval_scope']=args .test_retrieval_eval_scope
    CONFIG ['test_retrieval_first_expert_only']=args .test_retrieval_first_expert_only
    print ("--- 🚀 Initializing Test Run with Configuration ---")
    config_path =os .path .join (args .checkpoint_dir ,'training_config.pth')
    if os .path .exists (config_path ):
        print (f"Loading training config from {config_path } to match model...")
        saved_config =torch .load (config_path )
        CONFIG ['moe_settings']=saved_config ['moe_settings']
        CONFIG ['embedding_strategy']=saved_config ['embedding_strategy']
        CONFIG ['d_model']=saved_config ['d_model']
        CONFIG ['n_heads']=saved_config ['n_heads']
        CONFIG ['n_layers']=saved_config ['n_layers']
        CONFIG ['dropout']=saved_config ['dropout']
        CONFIG ['pretrained_settings']=saved_config .get ('pretrained_settings',CONFIG ['pretrained_settings'])
        CONFIG ['learned_settings']=saved_config .get ('learned_settings',CONFIG ['learned_settings'])
    else :
        print ("⚠️ No training config found, using default. This may cause state_dict mismatch.")
    log_input =args .test_log_name
    log_path_to_transform =None
    log_key_name =None
    log_path_obj =Path (log_input )
    if log_path_obj .exists ()and log_path_obj .is_file ():
        print (f"  - Test Log: Found direct path: {log_input }")
        log_path_to_transform =str (log_path_obj .resolve ())
        log_file_name =log_path_obj .name
        log_key_name =re .sub (r'\.xes(\.gz)?$','',log_file_name ,flags =re .IGNORECASE )
    else :
        print (f"  - Test Log: Looking up key in config: {log_input }")
        log_path_to_transform =CONFIG ['log_paths']['testing'].get (log_input )
        log_key_name =log_input
    if not log_path_to_transform :
        exit (f"❌ Error: Test log not found. '{log_input }' is not a valid path or config key.")
    if not Path (log_path_to_transform ).exists ():
        exit (f"❌ Error: Log file not found at resolved path: {log_path_to_transform }")
    print (f"  - Test Mode: {CONFIG ['test_mode']}")
    print (f"  - Test Episodes: {CONFIG ['num_test_episodes']}")
    print (f"  - Checkpoint Directory: {args .checkpoint_dir }")
    if args .checkpoint_epoch :
        print (f"  - Checkpoint Epoch: {args .checkpoint_epoch }")
    else :
        print ("  - Checkpoint Epoch: Latest")
    if CONFIG ['test_mode']=='retrieval_augmented':
        print (f"  - Retrieval K-values: {CONFIG ['test_retrieval_k']}")
        print (f"  - Retrieval Candidate %: {CONFIG ['test_retrieval_candidate_percentages']}")
        print (f"  - Retrieval Eval Scope: {CONFIG ['test_retrieval_eval_scope']}")
    strategy =CONFIG ['embedding_strategy']
    print (f"--- Running Testing Script in Stand-Alone Mode (strategy: '{strategy }') ---")
    device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
    print (f"Using device: {device }")
    checkpoint_dir =args .checkpoint_dir
    artifacts_path =os .path .join (checkpoint_dir ,'training_artifacts.pth')
    print ("\n📦 Loading test data...")
    loader =init_loader (CONFIG )
    loader .load_training_artifacts (artifacts_path )
    log_to_transform ={log_key_name :log_path_to_transform }
    print (f"Transforming log: '{log_key_name }' from {log_path_to_transform }")
    testing_logs =loader .transform (log_to_transform )
    torch .manual_seed (42 );
    np .random .seed (42 )
    model =create_model (CONFIG ,loader ,device )
    load_model_weights (
    model ,
    checkpoint_dir ,
    device ,
    epoch_num =args .checkpoint_epoch
    )
    unseen_log =testing_logs .get (log_key_name )
    if not unseen_log :
        exit (f"❌ Error: Test log '{log_key_name }' could not be processed.")
    print ("\n🛠️ Creating test tasks...")
    test_tasks ={
    'classification':get_task_data (unseen_log ,'classification'),
    'regression':get_task_data (unseen_log ,'regression')
    }
    test_mode =CONFIG .get ('test_mode','meta_learning')
    k_list_meta =CONFIG ['num_shots_test']
    k_list_retrieval =CONFIG .get ('test_retrieval_k',k_list_meta )
    if test_mode =='retrieval_augmented':
        print ("\n--- Running in Retrieval-Augmented Evaluation Mode ---")
        evaluate_retrieval_augmented (
        model ,
        test_tasks ,
        k_list_retrieval ,
        CONFIG ['num_test_episodes'],
        candidate_percentages =CONFIG .get ('test_retrieval_candidate_percentages'),
        first_expert_only =CONFIG .get ('test_retrieval_first_expert_only',False),
        eval_scope =CONFIG .get ('test_retrieval_eval_scope','experts')
        )
        print ("\n--- Running PCA-kNN Baseline Comparison ---")
        evaluate_pca_knn (
        model ,test_tasks ,k_list_retrieval ,CONFIG ['num_test_episodes']
        )
    elif test_mode =='meta_learning':
        print ("\n--- Running in Meta-Learning Evaluation Mode ---")
        evaluate_model (
        model ,test_tasks ,k_list_meta ,CONFIG ['num_test_episodes']
        )
    else :
        print (f"⚠️ Warning: Unknown test_mode '{test_mode }'. Defaulting to 'meta_learning'.")
        evaluate_model (
        model ,test_tasks ,k_list_meta ,CONFIG ['num_test_episodes']
        )
