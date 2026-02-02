import torch
import os
import re
from config import CONFIG
from data_generator import XESLogLoader
from components .meta_learner import MetaLearner
from components .moe_model import MoEModel
def init_loader (config ):
    strategy =config ['embedding_strategy']
    sbert_model_name =config ['pretrained_settings']['sbert_model']
    loader =XESLogLoader (strategy =strategy ,sbert_model_name =sbert_model_name )
    return loader
def create_model (config ,loader ,device ):
    strategy =config ['embedding_strategy']
    moe_config =config .get ('moe_settings',{})
    num_experts =moe_config .get ('num_experts',1 )
    if strategy =='pretrained':
        model_params ={'embedding_dim':config ['pretrained_settings']['embedding_dim']}
    else :
        if not loader .char_to_id :
            raise RuntimeError ("Loader must be fitted or artifacts loaded before creating 'learned' model.")
        model_params ={
        'char_vocab_size':len (loader .char_to_id ),
        'char_embedding_dim':config ['learned_settings']['char_embedding_dim'],
        'char_cnn_output_dim':config ['learned_settings']['char_cnn_output_dim'],
        }
    model =MoEModel (
    num_experts =num_experts ,
    strategy =strategy ,
    num_feat_dim =config ['num_numerical_features'],
    d_model =config ['d_model'],
    n_heads =config ['n_heads'],
    n_layers =config ['n_layers'],
    dropout =config ['dropout'],
    **model_params
    ).to (device )
    if strategy =='learned':
        model .set_char_vocab (loader .char_to_id )
    return model
def load_model_weights (model ,checkpoint_dir ,device ,epoch_num =None ):
    if not os .path .isdir (checkpoint_dir ):
        exit (f"❌ Error: Checkpoint directory not found at {checkpoint_dir }")
    checkpoint_path =None
    if epoch_num is not None :
        checkpoint_name =f"model_epoch_{epoch_num }.pth"
        checkpoint_path =os .path .join (checkpoint_dir ,checkpoint_name )
        if not os .path .exists (checkpoint_path ):
            exit (f"❌ Error: Specific checkpoint not found: {checkpoint_path }")
        print (f"🔍 Found specific checkpoint: {checkpoint_name }")
    else :
        checkpoints =[f for f in os .listdir (checkpoint_dir )if f .startswith ('model_epoch_')and f .endswith ('.pth')]
        if not checkpoints :
            exit (f"❌ Error: No model checkpoints found in {checkpoint_dir }.")
        latest_checkpoint_name =sorted (checkpoints ,key =lambda f :int (re .search (r'(\d+)',f ).group (1 )))[-1 ]
        checkpoint_path =os .path .join (
        checkpoint_dir ,
        latest_checkpoint_name
        )
        print (f"🔍 Found latest checkpoint: {latest_checkpoint_name }")
    print (f"💾 Loading weights from {checkpoint_path }...")
    model .load_state_dict (torch .load (checkpoint_path ,map_location =device ))
    return checkpoint_path
