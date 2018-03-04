#import the necessary libraries that are used in the program
import nltk
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class TextSimilarityExample:
    def __init__(self):
#statements in which we need to find similarity        
        self.statements = [
  'Thalapathi is a great movie',
  'Kamal hassan acted in many great movie',
  'Rajini is superstar',
  'Aamir khan is great actor']
#calculate term-frequency        
    def TF(self, sentence):
        words = nltk.word_tokenize(sentence.lower())
        freq = nltk.FreqDist(words)
        dictionary = {}
        for key in freq.keys():
            norm = freq[key]/float(len(words))
            dictionary[key] = norm
        return dictionary
#calculate inverse-document frequency    
    def IDF(self):
        def idf(TotalNumberOfDocuments, NumberOfDocumentsWithThisWord):
            return 1.0 + math.log(TotalNumberOfDocuments/NumberOfDocumentsWithThisWord)
        numDocuments = len(self.statements)
        uniqueWords = {}
        idfValues = {}
        for sentence in self.statements:
            for word in nltk.word_tokenize(sentence.lower()):
                if word not in uniqueWords:
                    uniqueWords[word] = 1
                else:
                    uniqueWords[word] += 1
        for word in uniqueWords:
            idfValues[word] = idf(numDocuments, uniqueWords[word])
        return idfValues
#TF multiplied by IDF for all the documents against a given search string
    def TF_IDF(self, query):
        words = nltk.word_tokenize(query.lower())
        idf = self.IDF()
        vectors = {}
        for sentence in self.statements:
            tf = self.TF(sentence)
            for word in words:
                tfv = tf[word] if word in tf else 0.0
                idfv = idf[word] if word in idf else 0.0
                mul = tfv * idfv
                if word not in vectors:
                    vectors[word] = []
                vectors[word].append(mul)
        return vectors
    def displayVectors(self, vectors):
        print(self.statements)
        for word in vectors:
            print("{} -> {}".format(word, vectors[word]))

    def cosineSimilarity(self):
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(self.statements)
        for j in range(1, 5):
            i = j - 1
            print("\tsimilarity of document {} with others".format(i))
            similarity = cosine_similarity(matrix[i:j], matrix)
            print(similarity)
    def demo(self):
        inputQuery = self.statements[0]
        vectors = self.TF_IDF(inputQuery)
        self.displayVectors(vectors)
        self.cosineSimilarity()

similarity = TextSimilarityExample()
similarity.demo()
