
import re
import numpy as np
from tokenizer_fun import TweetTokenizer

class TextPreprocessing:
    """
    This class takes a tweet as an input and returns padded word embeddings
    """

    def __init__(self, max_length_tweet=100, max_length_dictionary=2000000):

        """
        Initialize class
        """
        self.max_length_tweet = max_length_tweet
        self.max_length_dictionary = max_length_dictionary
        # import dictionary
        self.embeddings_dict = {}
        i = 0
        docs = ['glove_' + str(j) + '.txt' for j in range(1, 4)]

        for doc in docs:
            if i >= max_length_dictionary:
                break
            with open(doc, 'r') as file:
                for line in file:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    self.embeddings_dict[word] = vector
                    i += 1
                    if i >= max_length_dictionary:
                        break


        # import stopwords
        self.stopwords = []
        with open("english", 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                self.stopwords.append(word)

    def clean_text(self, tweet):

        """
        Clean text
        """

        # lower case
        tweet = tweet.lower()

        # remove links
        tweet = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", '', tweet)

        # replace # to ''
        tweet = re.sub(r"#", '', tweet)

        # remove numbers
        tweet = re.sub(r"[0-9]+", '', tweet)

        # remove stopwords and twitter handles
        tweet_cleaned_list = []
        for tweet_word in tweet.split(" "):
            if (tweet_word not in self.stopwords) & (~tweet_word.startswith('@')):
                tweet_cleaned_list.append(tweet_word)

        # join words
        tweet_cleaned = " ".join(tweet_cleaned_list)

        return tweet_cleaned

    @staticmethod
    def tokenize_text(tweet_cleaned):

        """
        Tokenize text
        """

        # define tokenizer
        tokenizer = TweetTokenizer()

        # tokenize
        tokenized_words = tokenizer.tokenize(tweet_cleaned)

        return tokenized_words

    def replace_token_with_index(self, tokenized):

        """
        Replace token with embeddings
        """

        word_embeddings = []
        for token in tokenized:
            embedding = self.embeddings_dict.get(token)
            if isinstance(embedding, np.ndarray):
                word_embeddings.append(embedding)

        return word_embeddings

    def pad_sequence(self, word_embeddings):

        """
        Pad embeddings for model
        """

        # if word_embeddings > max_length_tweet
        if len(word_embeddings) > self.max_length_tweet:
            word_embeddings_pad = word_embeddings[:self.max_length_tweet]

        # if word_embeddings == max_length_tweet
        elif len(word_embeddings) == self.max_length_tweet:
            word_embeddings_pad = word_embeddings

        # if word_embeddings < max_length_tweet
        else:
            pad = np.zeros_like(word_embeddings[0])
            diff = self.max_length_tweet - len(word_embeddings)
            word_embeddings.extend([pad] * diff)
            word_embeddings_pad = word_embeddings

        return word_embeddings_pad

    def process_tweet(self, tweet):

        """
        Run all functions defined above
        """

        clean = self.clean_text(tweet)
        tokenized = self.tokenize_text(clean)
        word_embeddings = self.replace_token_with_index(tokenized)
        word_embeddings_pad = self.pad_sequence(word_embeddings)

        return word_embeddings_pad