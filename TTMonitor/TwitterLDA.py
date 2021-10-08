import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .DocPooler import DocPooler

from nltk.tokenize.casual import TweetTokenizer
import gensim
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

def make_bigrams(phraser, doc_tokens):
    """
        Helper Method that takes a list of list of tokens (words). Each list of
        words represents one document. The phraser turns the list of tokens
        into a list of tokens enriched with bigrams. The output format is again
        a list of list of tokens where each list represents a document
    """

    return [phraser[tkn_list] for tkn_list in doc_tokens]

def get_top_topic(pred):
        """
        extracts the highest prediction score and the respective topic number
        """
        topic_number, score = zip(*pred)
        best = np.argmax(score)
        return((topic_number[best], score[best]))

class TwitterLDA:
    #TODO: get rid of tfidf vectorizer since it is only uses for tokenization   
    # Update: The tfidf vectorizer is needed for the calculation of the cosine similarity
    # since this needs a vector for each tweet/doc. However, one could and maybe should
    # use a standarized Tweet tokenizer to for the tokenizing part of the vecrotization
    def __init__(self, data, stop_words=None, tokenizer=TweetTokenizer(), enrich=True, doc_threshold=4, similarity_threshold=0.8):
        """
        self-doc_df: pd.DataFrame where each row represents a document. In the df are information about
                    the document, such as the hashtag on which it got created. A list of tweets_ids which
                    identify all tweets that are associated with that document. The full text of the 
                    document, which essentially is the concated collection of tweet texts
       
        self.tweet_df: A pd.DataFrame where each row represents a tweet. Here information such as tweet_id
                        tweet_text and geocoordinates are contained. Eventually also the most likely topic
                        and the respective score are contained. During the process all tweets that could
                        not be assigned to a topic during pooling are dropped.
        
        self.tokenizer: A nltk tokenizer which is used to transform a text into a list of tokens (words)

        enrich: determines if the documents should be enriched with tweets without hashtags, if they are
                somewhat similar to the document. A threshold for the similarity can be set

        self.phraser: a gensim phraser model that turns tokens into bigrams by default min_count=10, threshold=300
                        min_count (float, optional) – Ignore all words and bigrams with total collected count lower than this value.
                        threshold (float, optional) – Represent a score threshold for forming the phrases (higher means fewer phrases).
                        Heavily depends on concrete scoring-function, see the scoring parameter
        """
        if stop_words:
            self.stop_words=set(stop_words)
        else:
            self.stop_words = {}

        self.tokenizer = tokenizer#preserve_case, reduce_len, strip_handles

        #generate documents
        doc_pooler = DocPooler(data, doc_threshold = doc_threshold)
        print("Done with Pooling")
        if enrich:
            doc_pooler.enrich(similarity_threshold)
        self.doc_df,  self.tweet_df  = doc_pooler.docs, doc_pooler.assigned_tweets
        print("Done with Enriching")
        
        # list comprehension still twice as fast
        self.doc_tokens = list(self.tweet_token_gen(self.doc_df.full_text))

        print("Done with tokenizing")
        phrases = Phrases(self.doc_tokens, min_count=10, threshold=300)
        self.phraser = Phraser(phrases)
        print("Done with Initializing")
        
    
    def tweet_token_gen(self, data):
        """
        generator that takes the list of tweet dictionaries and returns the tokenized list of words
        It only uses the words that are also contained in the vocabulary of the tokenizer
        """
        #TODO: should probably be rather stopwords since vectorizer is only used for cosine similarity
        #vocab = self.vectorizer.vocabulary_
        
        for tweet in data:
            yield [token for token in self.tokenizer.tokenize(tweet) if token not in self.stop_words]
        
    def build_corpus(self, no_below, no_above):
        """
        takes a list of lists, which contains the tokens for each documents.
        For each document the tokens enriched with bigrams.
        min_count ignores bigrams but also words that are not rarer thatn mincount
        treshhold uses a scoring to determine the quality of a bigram and also does filtering
        """
        #https://radimrehurek.com/gensim/models/phrases.html
        #Prases is generating a Phrases object that automatically detect common phrases
        #Phraser is a object that actually dies the tranfromation ti a list of n-grams
        phraser = self.phraser
        doc_tokens = self.doc_tokens
        self.doc_bigrams = make_bigrams(phraser, doc_tokens)

        # reconstriucts the bigram bases on a id
        self.doc_id2bigram = corpora.Dictionary(self.doc_bigrams)
        self.doc_id2bigram.filter_extremes(keep_n=50000, no_below=no_below, no_above=no_above) #tuning param
        self.corpus_bi = [self.doc_id2bigram.doc2bow(doc_tkn) for doc_tkn in self.doc_bigrams]
        
    def get_topics(self, num_topics=30, num_words=15, log=False):
        return self.model.show_topics(num_topics=num_topics, num_words=num_words, log=log, formatted=False)
    
    def fit(self, n_topics=10, n_jobs=1, no_below=5, no_above=0.85,passes=100,chunksize=100):
        print("start fitting")
        self.build_corpus(no_below,no_above)
        self.coherence, self.n_topics, self.model = train_lda(self.corpus_bi, id2word=self.doc_id2bigram,
                                      num_topics=n_topics, texts=self.doc_bigrams,
                                      n_jobs= n_jobs, passes=passes)
        print("Done fitting")

    def classify_tweets(self):
        tok_tweets = list(self.tweet_token_gen(self.tweet_df.full_text))
        #tok_tweets = [[token for token in self.tokenizer.tokenize(tweet) ] for tweet in tqdm(self.tweets.full_text)] 
        #make bigrams
        tweet_bigrams = make_bigrams(self.phraser, tok_tweets)
        #transfom tweets according to gensim corpus
        ut_corpus_bi = [self.doc_id2bigram.doc2bow(tweet) for tweet in tweet_bigrams]
        
        predictions = self.model.get_document_topics(ut_corpus_bi, minimum_probability=0.0)#TODO:lookup
        topic_vec = [get_top_topic(pred) for pred in predictions]
        topics, scores = zip(*topic_vec)
        
        self.tweet_df["pred_topic"] = topics
        self.tweet_df["pred_score"] = scores
        self.tweet_df["predictions"]= [pred for pred in predictions]
        return topics, scores

    def save_to_disk(self, path):
        pickle.dump(( self.tweet_df, self.get_topics() ), open(path,"wb"))

def train_lda(corpus, id2word, num_topics, texts, n_jobs,passes=30,chunksize=100):
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, 
    #corpus — Stream of document vectors or sparse matrix of shape (num_terms, num_documents)
                                        id2word=id2word,
    #id2word – Mapping from word IDs to words. 
    #It is used to determine the vocabulary size, as well as for debugging and topic printing.
                                        num_topics=num_topics,
    #num_topics — The number of requested latent topics to be extracted from the training corpus.
                                        random_state=100,
    #random_state — Either a randomState object or a seed to generate one. Useful for reproducibility.
                                        #update_every=1,  
    #update_every — Number of documents to be iterated through for each update.
    #Set to 0 for batch learning, > 1 for online iterative learning.
    #NOT IN Lda.Multicore!!!
                                        chunksize=chunksize,
    #chunksize — Number of documents to be used in each training chunk.
                                        workers=n_jobs, 
    #workers: number of physical cpu-cores. use core-number - 1
                                        passes=passes,
    #passes — Number of passes through the corpus during training.
                                        #alpha='auto',
                                        per_word_topics=True)
    #per_word_topics — If True, the model also computes a list of topics, sorted in descending order of most
    #likely topics for each word, along with their phi values multiplied by the feature-length (i.e. word count)
    
    #calculate c_v coherence. see: https://radimrehurek.com/gensim/models/coherencemodel.html and http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf 
    coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=id2word, coherence='c_v')
    
    return (coherence_model_lda.get_coherence(), num_topics, lda)