import torch
import torch .nn .functional as F
import itertools
import numpy as np
from data_generator import XESLogLoader
try :
    from Levenshtein import distance as levenshtein_distance
except ImportError :
    levenshtein_distance =None
def evaluate_embedding_quality (model ,loader :XESLogLoader ):
    """
    Evaluates the quality of learned embeddings by comparing string distance
    (Levenshtein) to cosine similarity in the embedding space.
    """
    if model .strategy !='learned':
        return
    if levenshtein_distance is None :
        print ("\n⚠️ Skipping embedding evaluation: `pip install python-Levenshtein` to enable.")
        return
    print ("\n📊 Evaluating Learned Embedding Quality...")
    activity_names =loader .training_activity_names
    if len (activity_names )<2 :
        print ("  - Not enough activities in vocabulary to evaluate.")
        return
    with torch .no_grad ():
        model .eval ()
        embeddings =model .embedder .char_embedder (activity_names ,model .embedder .char_to_id )
        model .train ()
    embeddings =F .normalize (embeddings ,p =2 ,dim =1 ).cpu ().numpy ()
    pairs =[]
    for i ,j in itertools .combinations (range (len (activity_names )),2 ):
        name1 ,name2 =activity_names [i ],activity_names [j ]
        str_dist =levenshtein_distance (name1 ,name2 )/max (len (name1 ),len (name2 ))
        cos_sim =np .dot (embeddings [i ],embeddings [j ])
        pairs .append ({'str_dist':str_dist ,'cos_sim':cos_sim })
    if not pairs :
        return
    pairs .sort (key =lambda x :x ['str_dist'])
    num_pairs_to_show =min (5 ,len (pairs ))
    similar_by_name =pairs [:num_pairs_to_show ]
    dissimilar_by_name =pairs [-num_pairs_to_show :]
    avg_sim_for_similar_names =np .mean ([p ['cos_sim']for p in similar_by_name ])
    avg_sim_for_dissimilar_names =np .mean ([p ['cos_sim']for p in dissimilar_by_name ])
    print (f"  - Avg. Cosine Sim for Top {num_pairs_to_show } Similar Names:   {avg_sim_for_similar_names :.4f}")
    print (f"  - Avg. Cosine Sim for Top {num_pairs_to_show } Dissimilar Names: {avg_sim_for_dissimilar_names :.4f}")
    print ("-"*30 )
