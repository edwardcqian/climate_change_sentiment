from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import time
import datetime
import json

# twitter api keys
consumer_key = 'ckey'
consumer_secret = 'csecret'
access_token = 'atok'
access_token_secret = 'atoksecret'

# given a list of keywords and a tweet in string format, check if any keywords exist in the tweet
def check_keywords(keywords,tweet_str):
    t = False
    for astr in keywords:
        for x in astr.split():
            if x.lower() in tweet_str.lower():
                t = True
            else:
                t = False
                break
        if t == True:
            break
    return t

# load the tab seperated dictionary of query keywords
def load_qdict(file_path):
    qdict = dict()
    #Open the file
    with open(file_path,'r') as inf:
        for line in inf:
            sline = line.split('\t')
            if len(sline) > 2 or len(sline) < 2:
                continue
            if sline[1] not in qdict:
                qdict[sline[1]] = list()
                qdict[sline[1]].append(sline[0])
            else:
                qdict[sline[1]].append(sline[0])

    if not qdict:
        print("Dictionary of queries is empty.")
        exit()
    return qdict

# listener class for tweepy Stream
class JsonListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_data(self, data):
        try:
            dat = json.loads(data)
            if "extended_tweet" in dat:
                dat["extended_tweet"]["full_text"] = dat["extended_tweet"]["full_text"].replace("\"","$q$")
            else:
                dat['text'] = dat['text'].replace("\"","$q$")

            date = '_'.join([ dat['created_at'].split(' ')[i] for i in [1,2,5] ])

            # save tweet
            with open('data_'+date+'.txt', 'a') as file:
                json.dump(dat,file)
            for qk in qdict:
                queries = []
                for el in qdict[qk]:
                    queries.append(el)

                # save type
                if check_keywords(queries,data):
                    tmp_json = {'tweetid':dat['id'], 'date': date}
                    with open('stock_'+qk[:-1]+'.txt', 'a') as file:
                        json.dump(tmp_json,file)
        except BaseException as e:
            print('Failed on_data() ', str(e))
            print(dat)
            outf=open("error_log.txt","a")
            outf.write(str(datetime.datetime.now().isoformat())+"\t"+str(status)+"\n")
            outf.close()
            time.sleep(910)        

    def on_error(self, status):
        print('On Error! '+str(status))
        outf=open("error_log.txt","a")
        outf.write(str(datetime.datetime.now().isoformat())+"\t"+str(status)+"\n")
        outf.close()

        #if we have exceeded our rate limit
        if status == 420 or status == 429:
            #Wait 15 minutes before attempting to reconnect
            print("Rate Limit Exceeded.  Sleeping for 15 minutes.")
            time.sleep(901)
            return False

if __name__ == "__main__":
    qdict = load_qdict('my_queries.txt')

    queries = []
    for qk in qdict:
        for el in qdict[qk]:
            queries.append(el)
    print(queries)


    l = JsonListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l,tweet_mode='extended')
    stream.filter(track=queries)