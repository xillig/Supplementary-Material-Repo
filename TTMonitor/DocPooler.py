import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class DocPooler():
    """
    Data Structure for Pooling Tweets to Documents.
    This is done by creating a document for each unique hashtag and then assigning all the tweet texts of tweets that
    contain this hashtag.
    A tweet (text) can appear in multiple documents if it contains more then one hashtag. This overlapp
    is explicitly allowed.
    Since hashtags should only be considered a document if they occure in k different tweets
    a doc_threshold can be set to remove hashtags that occure in less tweets.

    Sometimes this still leads to very short docuemnts (i.e. docs that contain only a few tweets). In such cases
    it can be tried to enrich each documents with tweets that have not been assigned to a document yet if they
    are similar enough. For this purpose a similarity_treshold can be set in enrich(). All tweets that
    do not belong to a document 
    
    """
    def __init__(self,data_list, doc_threshold=5):
        self.doc_threshold = doc_threshold
        self.similarity_threshold = None

        #transform data to pd.Dataframe
        self.tweet_df = pd.DataFrame.from_dict({data["id_str"]:data for data in data_list}, orient="index")

        self.docs = self.tweet_df.explode("hashtags").groupby("hashtags").apply(self.create_documents)

        #apply doc_threshold
        if doc_threshold:
            self.docs = self.docs[self.docs.n_tweets > doc_threshold]
        self.update_assigned_tweets()
        


    def update_assigned_tweets(self):
        """
            Updates the tweets that are assigend to documents
            (e.g. after enriching or applying doc threshold)
        """
        assigned_tweets = self.docs.explode("tweet_ids").tweet_ids.unique()
        self.assigned_tweets = self.tweet_df.loc[assigned_tweets,:]

    def create_documents(self,grp):
        """
            pool documents based on hashtag. For each hashtag concat the text of all tweets
            which contain that tag. Also store the ids of the tweets that are assigned
            to a document.
        """
        doc_text = ' '.join(grp["full_text"])
        doc_ids = list(grp["id_str"].values)  
        return pd.Series({"full_text":doc_text, "tweet_ids":doc_ids,"n_tweets": len(doc_ids)})


    def enrich(self, sim_tresh=0.8):
        """ 
            for all the unassigned tweets this method calculates the cosine similiarity to all the documents
            if a document is more similar then sim_thresh, the tweet is assigned to the document.
            This is intentionally for growing small documents. In reality only works for very
            few documents and is quite expensive

        """
        self.similarity_threshold = sim_tresh
        # get the unique tweet ids that are associated with documents
        tweet_ids_of_docs = self.docs.explode("tweet_ids").tweet_ids.unique()
        unassiged_tweets = self.tweet_df.apply(lambda x: x["id_str"] not in tweet_ids_of_docs, axis=1)


        vectorizer = TfidfVectorizer(max_df=0.9 )
        doc_tokens = vectorizer.fit_transform(self.docs.full_text)
        hashtags = list(self.docs.index)
        tweets = self.tweet_df.loc[unassiged_tweets].full_text
        tweet_token = vectorizer.transform(tweets)

        for i,tag in enumerate(hashtags):
            #print(doc_tokens[i],tweet_token)
            to_keep = (cosine_similarity(doc_tokens[i] ,tweet_token) > sim_tresh)[0]
            to_keep = list(tweets[to_keep].index)
            
            for idx in to_keep:
                self.docs.at[tag,"full_text"] = self.docs.loc[tag,"full_text"] + tweets[idx]
                self.docs.at[tag,"tweet_ids"] = self.docs.loc[tag,"tweet_ids"] + [idx]
        
        self.docs["n_tweets"] = self.docs.tweet_ids.apply(len)
        self.update_assigned_tweets()
            #enrich
        
#if __name__ == "__main__":
#    import pandas as pd
#    import pickle
#    data_list = pickle.load(open("uk_data.pkl","rb"))
#    from DocPooler import DocPooler
#    d1 = DocPooler(data_list)
#    d1.enrich()
