"""
References:
Jannach, Dietmar, and Malte Ludewig. "When recurrent neural networks meet the neighborhood for session-based recommendation." Proceedings of the Eleventh ACM Conference on Recommender Systems. 2017.
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
"""

import os
import time
import numpy as np
import pickle as pkl
import pandas as pd
from _datetime import timezone, datetime
import gc

import vmknn_ml6 as vsknnH
#from hybrid import weighted as wh
import evaluation_ml6 as eval
import accuracy_ml6 as ac

def load_data( path, file_name):
    '''
    Desc: Loads a tuple of training and test set with the given parameters. 
    --------
    Input:
        path : Path of preprocessed data (str)
        file : Name of dataset
    --------
    Output : tuple of DataFrame (train, test)
    '''
    
    print('START load data') 
    st = time.time()
    sc = time.perf_counter()
        
    train_appendix = '_train'
    test_appendix = '_test'
                
    train = pd.read_csv(path + file_name + train_appendix +'.txt', sep='\t', dtype={'ItemId':np.int64})
    test = pd.read_csv(path + file_name + test_appendix +'.txt', sep='\t', dtype={'ItemId':np.int64} )
          
    data_start = datetime.fromtimestamp( train.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( train.Time.max(), timezone.utc )
    
    print('train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(train), train.SessionId.nunique(), train.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    data_start = datetime.fromtimestamp( test.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( test.Time.max(), timezone.utc )
    
    print('test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n'.
          format( len(test), test.SessionId.nunique(), test.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    
    print( 'END load data ', (time.perf_counter()-sc), 'c / ', (time.time()-st), 's' )
    
    return (train, test)


if __name__ == '__main__':
   
    # read pickle of article embeddings
    os.chdir('/home/alireza/Desktop/ml6/session_based/')
    filename = 'article_embeddings.pickle'
    infile = open(filename,'rb')
    content = pkl.load(infile)[2]
    # for now we drop articles that we dont have their metadata, so we also drop the embedding of padding article
    content = np.delete(content, 0, axis=0)

    # read the preprocessed data
#    os.chdir('../')
    data_path = '/home/alireza/Desktop/ml6/session_based/'
    file_prefix = 'newdeal'
            
    # create a list of metric classes to be evaluated
    metric = []
    
    metric.append(ac.Precision(20))
    metric.append(ac.Precision(10))
    metric.append(ac.Precision(5))
    metric.append(ac.Recall(20))
    metric.append(ac.Recall(10))
    metric.append(ac.Recall(5))
    metric.append( ac.Diversity(20) )
    metric.append( ac.Diversity(10) )
    metric.append( ac.Diversity(5) )   
    metric.append( ac.DiversityRankRelavance(20) )
    metric.append( ac.DiversityRankRelavance(10) )
    metric.append( ac.DiversityRankRelavance(5) )

    # predictor
    vknna = vsknnH.VMContextKNN( 100, 2000, last_n_days=None, extend=False )
    
    # Hybrid example
#    hybrid = wh.WeightedHybrid( [sknnH.ContextKNN( 100, 500, similarity="cosine", extend=False ), CB.CB()], [0.7,0.3], fit=True )

    # load data        
    print('data_path: ', data_path)
    
    train, test = load_data(data_path, file_prefix)
    item_ids = train.ItemId.unique()

    # train algorithms
    ts = time.time()
    print('start fiting')
    vknna.fit(train,content)
    print(' time: ', (time.time() - ts))

    # for only prediction (real case recommendation) we can use predict_next, we should drop the events of the query user_id from data
#    vknna.predict_next(user_id, input_item_id, item_set, predict_for_item_ids, timestamp=ts)
    
#    # init metrics (for evaluation)
#    for m in metric:
#        m.init(content)
#     
#    # evaluation
#    result = eval.evaluate_sessions(vknna, metric, test, train)
#  
#    # print results
#    for e in result:
#        print( e[0], ' ', e[1])
#      
#    del metric