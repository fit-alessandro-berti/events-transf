import torch
import torch .nn as nn
import math
class PositionalEncoding (nn .Module ):
    def __init__ (self ,d_model ,dropout =0.1 ,max_len =512 ):
        super ().__init__ ()
        self .dropout =nn .Dropout (p =dropout )
        position =torch .arange (max_len ).unsqueeze (1 )
        div_term =torch .exp (torch .arange (0 ,d_model ,2 )*(-math .log (10000.0 )/d_model ))
        pe =torch .zeros (1 ,max_len ,d_model )
        pe [0 ,:,0 ::2 ]=torch .sin (position *div_term )
        pe [0 ,:,1 ::2 ]=torch .cos (position *div_term )
        self .register_buffer ('pe',pe )
    def forward (self ,x ):
        x =x +self .pe [:,:x .size (1 ),:]
        return self .dropout (x )
class EventEncoder (nn .Module ):
    def __init__ (self ,d_model ,n_heads ,n_layers ,dropout =0.1 ):
        super ().__init__ ()
        self .pos_encoder =PositionalEncoding (d_model ,dropout )
        self .d_model =d_model
        self .cls_token =nn .Parameter (torch .zeros (1 ,1 ,d_model ))
        nn .init .normal_ (self .cls_token ,std =0.02 )
        encoder_layer =nn .TransformerEncoderLayer (
        d_model =d_model ,
        nhead =n_heads ,
        dim_feedforward =d_model *4 ,
        dropout =dropout ,
        batch_first =True ,
        activation ='gelu',
        norm_first =True
        )
        self .transformer_encoder =nn .TransformerEncoder (encoder_layer ,num_layers =n_layers )
        self .pool_query =nn .Parameter (torch .zeros (1 ,1 ,d_model ))
        nn .init .normal_ (self .pool_query ,std =0.02 )
        self .mha_pool =nn .MultiheadAttention (
        embed_dim =d_model ,
        num_heads =n_heads ,
        dropout =dropout ,
        batch_first =True
        )
        self .final_projection =nn .Linear (d_model *2 ,d_model )
        self .out_norm =nn .LayerNorm (d_model )
    def forward (self ,src ,src_key_padding_mask =None ):
        B ,T ,D =src .shape
        cls =self .cls_token .expand (B ,1 ,D )
        src =torch .cat ([cls ,src ],dim =1 )
        if src_key_padding_mask is not None :
            pad_col =torch .zeros ((B ,1 ),dtype =torch .bool ,device =src_key_padding_mask .device )
            src_key_padding_mask =torch .cat ([pad_col ,src_key_padding_mask ],dim =1 )
        src =src *math .sqrt (self .d_model )
        src =self .pos_encoder (src )
        output =self .transformer_encoder (src ,src_key_padding_mask =src_key_padding_mask )
        cls_out =output [:,0 ,:]
        tokens =output [:,1 :,:]
        token_mask =None
        if src_key_padding_mask is not None :
            token_mask =src_key_padding_mask [:,1 :]
        pool_q =self .pool_query .expand (B ,-1 ,-1 )
        pooled_out ,_ =self .mha_pool (
        query =pool_q ,
        key =tokens ,
        value =tokens ,
        key_padding_mask =token_mask
        )
        pooled =pooled_out .squeeze (1 )
        concatenated =torch .cat ([cls_out ,pooled ],dim =-1 )
        projected =self .final_projection (concatenated )
        encoded =self .out_norm (projected )
        return encoded
