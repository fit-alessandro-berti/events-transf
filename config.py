import os
LOG_DIR ='./logs'
CONFIG ={
'log_paths':{
'training':{
'o2c':os .path .join (LOG_DIR ,'0001_o2c.xes.gz'),
'hire2retire':os .path .join (LOG_DIR ,'00002_hire2retire.xes.gz'),
'di2re':os .path .join (LOG_DIR ,'00003_di2re.xes.gz'),
'mak2stock':os .path .join (LOG_DIR ,'00004_mak2stock.xes.gz'),
'offer2accept':os .path .join (LOG_DIR ,'00005_offer2accept.xes.gz'),
'quote2order':os .path .join (LOG_DIR ,'00006_quote2order.xes.gz'),
'opp2quote':os .path .join (LOG_DIR ,'00007_opp2quote.xes.gz'),
'lead2opp':os .path .join (LOG_DIR ,'00008_lead2opp.xes.gz'),
'p2p':os .path .join (LOG_DIR ,'00009_p2p.xes.gz'),
'rid2mit':os .path .join (LOG_DIR ,'00010_rid2mit.xes.gz'),
'req2receipt':os .path .join (LOG_DIR ,'00011_req2receipt.xes.gz'),
},
'testing':{
'D_unseen':os .path .join (LOG_DIR ,'00013_clos2rep.xes.gz')
}
},
'moe_settings':{
'num_experts':4
},
'embedding_strategy':'learned',
'pretrained_settings':{
'sbert_model':'all-mpnet-base-v2',
'embedding_dim':768 ,
},
'learned_settings':{
'char_embedding_dim':64 ,
'char_cnn_output_dim':128 ,
},
'd_model':256 ,
'n_heads':8 ,
'n_layers':6 ,
'dropout':0.15 ,
'num_numerical_features':3 ,
'num_shots_range':(1 ,20 ),
'num_queries':10 ,
'num_shots_test':[1 ,5 ,10 ,20 ],
'lr':1e-4 ,
'epochs':10 ,
'episodes_per_epoch':1000 ,
'episodic_label_shuffle':'mixed',
'training_strategy':'mixed', #episodic, retrieval, mixed
'retrieval_train_k':5 ,
'retrieval_train_batch_size':64 ,
'test_mode':'retrieval_augmented',
'test_retrieval_k':[1 ,5 ,10 ,20 ],
'test_retrieval_candidate_percentages':[0.5 ,1 ,3 ,5 ],
'test_retrieval_first_expert_only':True ,
'num_test_episodes':200 ,
'num_cases_for_testing':500 ,
}
if False :
    OUT_DIR =os .path .join (LOG_DIR ,'out')
    if os .path .isdir (OUT_DIR ):
        replaced_training_logs =False
        if os .listdir (OUT_DIR ):
            CONFIG ['log_paths']['training']={}
            replaced_training_logs =True
        out_logs =[
        name for name in os .listdir (OUT_DIR )
        if (name .startswith ('log_')or name .startswith ('simulated_'))
        and os .path .isfile (os .path .join (OUT_DIR ,name ))
        ]
        for name in sorted (out_logs ):
            CONFIG ['log_paths']['training'][name ]=os .path .join (OUT_DIR ,name )
        if replaced_training_logs :
            print ('Replaced training logs with logs from:',OUT_DIR )
            print ('Training logs:',CONFIG ['log_paths']['training'])
