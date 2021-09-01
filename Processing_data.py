nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(STOPWORDS)

# Remove stopwords
custom_stopwords = ['hi', '\n', '\n\n', '&amp;', ' ', '.', '-', 'got', "it's", 'it’s', "i'm", 'i’m', 'im', 'want',
                    'like', '$', '@']

# Remove stop words by adding to the default list
nlp = English()
STOP_WORDS = nlp.Defaults.stop_words.union(custom_stopwords)
# ALL_STOP_WORDS = spacy + gensim + wordcloud
ALL_STOP_WORDS = STOP_WORDS.union(SW).union(stopwords)

def get_only_words(wordsList):
    pattern = re.compile('[a-zA-Z]+')
    result = []
    for word in wordsList:
        if pattern.match(word) != None:
            result.append(word)
    return result


def get_words_without_stop(sentence):
    sentence = sentence.lower()
    words = word_tokenize(sentence)
    withoutstopwords = [word for word in words if not word in ALL_STOP_WORDS]
    return get_only_words(withoutstopwords)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

vectorizer = TfidfVectorizer(tokenizer=get_words_without_stop)
vectorizedTrainData = vectorizer.fit_transform(trainContent)
vectorizedTestData = vectorizer.transform(testContent)

from sklearn.preprocessing import MultiLabelBinarizer
multiLabelBinary = MultiLabelBinarizer()
trainBinaryLabel = multiLabelBinary.fit_transform(labelTrain)
#y=list(multiLabelBinary.classes_)
testBinaryLabel = multiLabelBinary.transform(labelTest)
