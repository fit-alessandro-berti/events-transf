import torch
import torch .nn as nn
import pandas as pd
from .char_cnn_embedder import CharCNNEmbedder
class LearnedEventEmbedder (nn .Module ):
    def __init__ (self ,char_vocab_size :int ,char_emb_dim :int ,char_cnn_out_dim :int ,
    num_feat_dim :int ,d_model :int ,dropout :float =0.1 ):
        super ().__init__ ()
        self .char_embedder =CharCNNEmbedder (
        char_vocab_size ,char_emb_dim ,char_cnn_out_dim
        )
        self .char_to_id ={}
        total_input_dim =(2 *char_cnn_out_dim )+num_feat_dim
        self .projection =nn .Sequential (
        nn .LayerNorm (total_input_dim ),
        nn .Linear (total_input_dim ,d_model ),
        nn .GELU (),
        nn .LayerNorm (d_model )
        )
        self .dropout =nn .Dropout (dropout )
    def forward (self ,events_df :pd .DataFrame ):
        activity_names =events_df ['activity_name'].tolist ()
        resource_names =events_df ['resource_name'].tolist ()
        act_emb =self .char_embedder (activity_names ,self .char_to_id )
        res_emb =self .char_embedder (resource_names ,self .char_to_id )
        device =act_emb .device
        num_arr =events_df [['cost','time_from_start','time_from_previous']].values
        num_feats =torch .log1p (torch .as_tensor (num_arr ,dtype =torch .float32 ,device =device ).clamp_min (0 ))
        combined_input =torch .cat ([act_emb ,res_emb ,num_feats ],dim =-1 )
        return self .dropout (self .projection (combined_input ))
