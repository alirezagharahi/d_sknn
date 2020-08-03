from _operator import itemgetter
from math import sqrt
import math
import random
import time
import numpy as np
import pandas as pd
import os
import psutil
import gc
from sklearn.metrics.pairwise import cosine_similarity
import operator
from statistics import mean,stdev

class ContextKNN:
    '''
    ContextKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''
    
    def __init__( self, k, sample_size=1000, sampling='recent',  similarity = 'jaccard', content_similarity = 'jaccard', remind=False, pop_boost=0, extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time'):
       
        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.content_similarity = content_similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_variance = dict()
        self.session_diversity_map = dict()
        self.session_time = dict()
        self.sim_time = 0
        
    def content_aggregator (self, content, indexes):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
        embeddings = content[indexes]
        aggregate = np.mean(embeddings, axis=0)
        
        return aggregate    

    def vector_variance (self, content, indexes):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
        diags_cov_var_matrix=np.cov(np.transpose(content[indexes])).diagonal()
        return diags_cov_var_matrix.sum()     
    
    def content_Weighted_aggregator (self, content, indexes, map):
        '''
        content: input from article emdeddings containing labelencoders, dataframe and embeddings
        indexes: article ids in the session
        '''
        weight = []
        for item in map:
            weight.append(map[item])
        embeddings = content[indexes]
        aggregate = np.average(embeddings,weights=weight, axis=0)
        
        return aggregate 
        
    
    def fit(self, train, content, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        
        content: It contains article embeddings for each article in np.array format (the index of array can be mapped with LabelEncoder for articles)
        
        the session article_id column should also be changed by article labelEncoder
        '''
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        index_time = train.columns.get_loc( self.time_key )
        

        self.content = content

        session = -1
        session_items = set()

        time = -1

        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session : session_items})
                    self.session_diversity_map.update({session : self.diversity_of_session(session)})
                    self.session_time.update({session : time})

                session = row[index_session]
                session_items = set()

            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get( row[index_item] )
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item] : map_is})
            map_is.add(row[index_session])
            
        # Add the last tuple    
        self.session_time.update({session : time})
        self.session_item_map.update({session : session_items})
        self.session_diversity_map.update({session : self.diversity_of_session(session)})
        
        self.min_div = min(self.session_diversity_map.values())
        self.max_div = max(self.session_diversity_map.values())
        
        self.session_diversity_map = {k: self.diversity_of_session(k) for k, v in self.session_item_map.items()}

    def predict_next( self, session_id, input_item_id, item_set, predict_for_item_ids, skip=False, type='view', timestamp=0 ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        
        if( self.session != session_id ): #new session
            
            if( self.extend ):
                item_set = set( self.session_items )
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get( item )
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item : map_is})
                    map_is.add(self.session)
                    
                ts = time.time()
                self.session_time.update({self.session : ts})
                
                
            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()
        
        if type == 'view':
            self.session_items.append( input_item_id )
        
        if skip:
            return
        
        self.session_item_map.update({session_id : set(item_set)})
        self.session_diversity_map.update({session_id : self.diversity_of_session(session_id)})
        
        neighbors = self.find_neighbors( set(item_set), input_item_id, session_id)
        scores = self.score_items( neighbors , item_set )
                
        # add some reminders
        if self.remind:
             
            reminderScore = 5
            takeLastN = 3
             
            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                 
                oldScore = scores.get( elem )
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt/100)
                scores.update({elem : newScore})
        
        #push popular ones
        if self.pop_boost > 0:
               
            pop = self.item_pop( neighbors )
            # Iterate over the item neighbors
            #print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + (self.pop_boost * item_pop))})
         
        
        # Create things in the format ..
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
        
        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)
        
        if self.normalize:
            series = series / series.max()
        return series 

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session( session )
            for item in items:
                
                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})
                    
                if( result.get(item) > max_pop ):
                    max_pop =  result.get(item)
         
        for key in result:
            result.update({key: ( result[key] / max_pop )})
                   
        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        return random.random()
    

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id )
    
    def variance_for_session(self, session_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_variance.get( session_id )
        
    def diversity_for_session(self, session_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_diversity_map.get( session_id )    
    
    def diversity_of_session(self, session):
        '''
        Returns artist diversity of a session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        session_items = list(self.items_for_session(session))
        dis = 0.0
        pairs = 0.0
        
        for i in range(len(session_items)):
            contentA = self.content[ session_items[i],: ]
            for j in range(i+1,len(session_items)):
                contentB = self.content[ session_items[j],:]
                dis += self.cos_Dis_sim(contentA,contentB)
                pairs += 1.0
        
        diversity = dis / pairs if pairs > 0 else 0
        return diversity
    
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else: #sample some sessions
                
            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item( input_item_id );
                         
            if len(self.relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( self.relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( self.relevant_sessions, self.sample_size )
                else:
                    sample = self.relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return self.relevant_sessions
                        
    def calc_similarity (self, content, session_items, sessions, diversity=False ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric

        neighbors = []
        cnt = 0
        #-----------------
        if diversity is False:
        #-----------------    
            for session in sessions:
                cnt = cnt + 1
                # get items of the session, look up the cache first 
                session_items_test = self.items_for_session( session ) 
                similarity = getattr(self , self.similarity)(session_items_test, session_items)
                if similarity > 0 :
                    neighbors.append((session, similarity))

        else:
            for session in sessions:
                cnt = cnt + 1
                # get items of the session, look up the cache first 
                session_items_test = self.items_for_session( session ) 
                similarity = getattr(self , self.similarity)(session_items_test, session_items)
                if similarity > 0:
                    neighbors.append((session, similarity,(self.diversity_for_session(session))))

        return neighbors

    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items, input_item_id, session_id )
        possible_neighbors = self.calc_similarity(self.content, session_items, possible_neighbors , diversity=True)
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors, current_session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        current_session: items in the current sessions
        
        Returns
        --------
        out : list of tuple (item, score)           
        '''
        current_session_embedding = self.content_aggregator(self.content,list(current_session))
        
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        values = []
        for session in neighbors:
            # get the items in this session
            values.append(session[2])
            items = self.items_for_session(session[0])
            diversity = session[2] 
            
            for item in items:
                old_score = scores.get( item )                
                new_score = session[1]*diversity
                   
                if old_score is None:        
                    scores.update({item : ( new_score * diversity)})

                else: 
                    new_score = old_score + new_score*diversity
                    scores.update({item : new_score})
     
        for w, v in scores.items():
             item_embedding = self.content[w,:]
             scores[w] = v * (self.cos_Dis_sim(item_embedding,current_session_embedding))  # Apply a weight for diversity          

        return scores
    
    def cos_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return ((dot_product / (norm_a * norm_b))+1)/2
    
    def cos_Dis_sim(self,a,b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return (1-(dot_product / (norm_a * norm_b)))/2