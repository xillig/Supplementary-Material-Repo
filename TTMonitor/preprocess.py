from datetime import datetime
import glob
import json
import numpy as np
import os
import pandas as pd
import re

import sys
from tqdm.notebook import tqdm


def read_datafiles(directory):
    json_data = []
    
    for filename in tqdm(glob.glob(directory+'/*.json')):
        try:
            with open(filename,'r') as f:
                for line in f:

                    try: #check if a json-object is complete ( "}" will occur to close the "{", json.loads(.) wont throw an error. )
                        tweet = json.loads(line)
                        #json_data.append(json.loads(line))
                        result = dict()
                    except ValueError:
                        # Not yet a complete JSON value
                        continue
                                     

                    if any([tweet.get("is_quote_status"),
                            tweet.get("retweeted"),
                            tweet.get("in_reply_to_status_id_str")]):
                        continue
                    
                    extended_tweet  = tweet.pop("extended_tweet", None)
                    if not extended_tweet:
                        continue   
                    
                    user = tweet.pop("user",None)
                    entities = extended_tweet.pop("entities",None)
                        
                    place = tweet.pop("place", None)
                    if not place:
                        continue
                    bbox = place.get("bounding_box")
                    coordinates = [co for co in bbox.get("coordinates")]
                    
 
                    #packing result
                    result["created_at"] = tweet.get("created_at")
                    result["id_str"] = tweet.get("id_str")
                    #result["in_reply_to_status_id_str"] = tweet.get("in_reply_to_status_id_str")
                    #result["is_quote_status"] = tweet.get("is_quote_status")
                    #result["retweeted_status"] = tweet.get("retweeted_status")
                
                    result["user_id_str"] = user.get("id_str")
                    result["full_text"] = extended_tweet.get("full_text")
                    result["hashtags"] = [tag["text"] for tag in entities.get("hashtags")]
                    result["lang"] = tweet.get("lang")
                    result["place_full_name"] = place.get("full_name")
                    result["country_code"] = place.get("country_code")
                    result["coordinates"] = coordinates
                    x_mean = np.mean([x for pair in coordinates for x,y in pair ])
                    y_mean = np.mean([y for pair in coordinates for x,y in pair ])
                    result["center_coord_X"] = x_mean
                    result["center_coord_Y"] = y_mean
                     
                    
                        
                    json_data.append(result)
        except Error:
            raise Error
            
            next
    print("Number of Tweets",len(json_data))
    print("Size of Data",sys.getsizeof(json_data)/(1e+9), "GB")
    return json_data


#getting the indexes, to check sub-json 'framed_data['place']':
#check is location is availiable and if the Tweets originate from the USA:

def parse_tweets(tweets, include_hashtags=False):
    """
    Each tweet gets parsed. This include:
        Removal of hyperlinks
        text to lower
        removes some (but not all) emojis
        removes mentions
        if include_hashtags == True only the # symbol will be stripped.
        else the whole hashtag will be remove. If the hashtags are kept
        they are guaranteed to have a high frequency in documents
        since every tweets in the doc has the hashtag for sure

    """
    def parse_tweet(tweet, include_hashtags = False):
        #remove url
        tweet["full_text"] =  re.sub("https://t[.]co/[\S]+","",tweet["full_text"])
        tweet["full_text"] = tweet["full_text"].lower()

        #emojis = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])') #emoji unicode
        #tweet["emojis"] = emojis.findall(tweet["full_text"])
        #tweet["full_text"] = re.sub(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])',
        #       "", tweet["full_text"])

        tweet["full_text"] = re.sub("@[\S]+","", tweet["full_text"])
        if include_hashtags:
            #remove only '#' symbols
            tweet["full_text"] = re.sub("#"," ", tweet["full_text"])
        else:
            #remove full_hashtag
            tweet["full_text"] = re.sub("#[\S]+","", tweet["full_text"])
        return tweet
    
    tweets = [parse_tweet(tweet, include_hashtags=include_hashtags) for tweet in tweets]
    return tweets


def filter_tweets(tweets, keywords, include = True):
    """
        Filters tweets based on a keyword list. By default if any keyword is contained in the (lowered) tweet text
        the tweet is preserved. All the other tweets are dropped.
        Setting 'include' to False will perform an exclusive filter. If any keyword is contained, a tweet is excluded.
    
    """
    def keyword_filter(full_text, keywords, include = include):
        keywords = set([word.lower() for word in keywords])
        tweet_words = set(full_text.split())
        contains_keyword = not keywords.isdisjoint(tweet_words)
        if include:
            return contains_keyword
        else:
            return not contains_keyword
    return [tweet for tweet in tqdm(tweets) if keyword_filter(tweet["full_text"], keywords)]


