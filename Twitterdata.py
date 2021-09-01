import tweepy
# API information
consumer_key = "X"
consumer_key_secret = "XX"
access_token = "XXX"
access_token_secret = "XXXX"

# Authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)

# Set access to user's access key and access secret
auth.set_access_token(access_token, access_token_secret)

# Calling the API
# api =tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

def limit_handled(cursor):
  while True:
    try:
        yield cursor.next()
    except StopIteration:
            return
    except tweepy.TweepError:
        time.sleep(15 * 60)

topics = ['politics','politics and travel','politics and car','electronics and car','electronics','car','travel','travel and electronics','travel and car']
if __name__ == '__main__':
  for topic in topics:
    users = {}
    for tweet in limit_handled(tweepy.Cursor(api.search, q=(topic), count=100, lang='en').items(5000)):
      print(tweet.user.name)
      print(tweet.text)
      if tweet.user.id in users:
        users[tweet.user.id].append(tweet.text)
      else:
        users[tweet.user.id] = []
        users[tweet.user.id].append(tweet.text)
    # Generate filename with twitter data - TwitterData
    path = os.getcwd()
    fileName =  path + '/drive/MyDrive/Data/TwitterData/' + topic + '_train_data.data'
    print (fileName)
    userTweets = open(fileName,'ab+')
    pickle.dump(users,userTweets)
    userTweets.close()

    fileName = ['electronics_train_data.data', 'politics_train_data.data', 'travel_train_data.data',
                'car_train_data.data',
                'electronics and car_train_data.data', 'politics and car_train_data.data',
                'travel and car_train_data.data',
                'politics and travel_train_data.data', 'travel and electronics_train_data.data']

    labelsFile = [['electronics'], ['politics'], ['travel'], ['car'],
                  ['electronics', 'car'], ['politics', 'car'], ['travel', 'car'],
                  ['politics', 'travel'], ['travel', 'electronics']]


    def get_combine_tweets_per_user(tweetList):
        result = ''
        for tweet in tweetList:
            result = result + ' '
            result = result + tweet
        return result


    def read_pickle_files():
        labelData = []
        contentData = []
        for index in range(len(fileName)):
            pkl_file = open('/content/drive/MyDrive/Data/TwitterData/' + fileName[index], 'rb')
            getDict = pickle.load(pkl_file)
            for user in getDict:
                aggregatedDoc = get_combine_tweets_per_user(getDict[user])
                contentData.append(aggregatedDoc)
                labelData.append(labelsFile[index])
        return labelData, contentData


    label, Content = read_pickle_files()

    from sklearn.model_selection import train_test_split

    trainContent, testContent, labelTrain, labelTest = train_test_split(Content, label, test_size=.2, random_state=1)
