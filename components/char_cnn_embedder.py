import torch
import torch .nn as nn
import torch .nn .functional as F
import numpy as np
class CharCNNEmbedder (nn .Module ):
    def __init__ (self ,char_vocab_size :int ,char_embedding_dim :int ,output_dim :int ,max_word_len :int =30 ):
        super ().__init__ ()
        self .max_word_len =max_word_len
        self .char_embedding =nn .Embedding (char_vocab_size ,char_embedding_dim ,padding_idx =0 )
        self .convs =nn .ModuleList ([
        nn .Conv1d (in_channels =char_embedding_dim ,out_channels =32 ,kernel_size =3 ,padding =1 ),
        nn .Conv1d (in_channels =char_embedding_dim ,out_channels =32 ,kernel_size =4 ,padding =2 ),
        nn .Conv1d (in_channels =char_embedding_dim ,out_channels =32 ,kernel_size =5 ,padding =2 ),
        ])
        cnn_output_dim =32 *len (self .convs )
        self .projection =nn .Sequential (
        nn .Linear (cnn_output_dim ,output_dim ),
        nn .LayerNorm (output_dim )
        )
    def _strings_to_char_ids (self ,strings :list [str ],char_to_id :dict ):
        device =self .char_embedding .weight .device
        batch_char_ids =[]
        unk_id =char_to_id .get ('<UNK>',1 )
        for s in strings :
            s =s [:self .max_word_len ]
            char_ids =[char_to_id .get (c ,unk_id )for c in s ]
            padded_ids =char_ids +[0 ]*(self .max_word_len -len (char_ids ))
            batch_char_ids .append (padded_ids )
        return torch .tensor (batch_char_ids ,dtype =torch .long ,device =device )
    def forward (self ,strings :list [str ],char_to_id :dict ):
        char_ids =self ._strings_to_char_ids (strings ,char_to_id )
        embedded_chars =self .char_embedding (char_ids )
        embedded_chars =embedded_chars .permute (0 ,2 ,1 )
        conv_outputs =[F .gelu (conv (embedded_chars ))for conv in self .convs ]
        pooled_outputs =[F .max_pool1d (out ,out .shape [2 ]).squeeze (2 )for out in conv_outputs ]
        concatenated =torch .cat (pooled_outputs ,dim =1 )
        final_embedding =self .projection (concatenated )
        return final_embedding
