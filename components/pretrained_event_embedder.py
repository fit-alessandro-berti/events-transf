import torch
import torch .nn as nn
import numpy as np
import pandas as pd
class PretrainedEventEmbedder (nn .Module ):
    def __init__ (self ,embedding_dim :int ,num_feat_dim :int ,d_model :int ,dropout :float =0.1 ):
        super ().__init__ ()
        total_input_dim =embedding_dim +num_feat_dim
        self .projection =nn .Sequential (
        nn .LayerNorm (total_input_dim ),
        nn .Linear (total_input_dim ,d_model ),
        nn .GELU (),
        nn .LayerNorm (d_model )
        )
        self .dropout =nn .Dropout (dropout )
    def forward (self ,events_df :pd .DataFrame ):
        device =next (self .parameters ()).device
        act_emb =torch .from_numpy (np .stack (events_df ['activity_embedding'].values )).float ().to (device )
        res_emb =torch .from_numpy (np .stack (events_df ['resource_embedding'].values )).float ().to (device )
        semantic_emb =act_emb +res_emb
        num_arr =events_df [['cost','time_from_start','time_from_previous']].values
        num_feats =torch .log1p (torch .as_tensor (num_arr ,dtype =torch .float32 ,device =device ).clamp_min (0 ))
        combined_input =torch .cat ([semantic_emb ,num_feats ],dim =-1 )
        return self .dropout (self .projection (combined_input ))
