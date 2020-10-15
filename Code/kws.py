import nltk
import fileUtil
import freqUtil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import Tree
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import math

#NP CHUNKING?
#LEMMATIZING???
#LOCATIONS???

#COULD CHUNK PATTERNS LIKE (DT? + NN + IN + NN)
#('the', 'DT')
#('god', 'NN')
#('of', 'IN')
#('thunder', 'NN')
#TODO: Setup GIT repo.

sent_detector = nltk.data.load('nltk_data/tokenizers/punkt/english.pickle')
stopwords_eng = stopwords.words('english')
punctuation = ["@", "'", "'s", ".", ",", "?", "!", "(", ")", "[", "]", "{", "}"]

idfs = {}

def getNgrams(tokens, n):
    ngrams = []
    for i in range(0, len(tokens)-n+1):
        ngrams.append(' '.join(tokens[i: i+n]))
    print(ngrams)
    return ngrams


def getFrequencyCounts(filename):
    content = fileUtil.readData(filename)

    all_tokens = []

    sentences = sent_detector.tokenize(content.strip())
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        #print(token)

        tokens = word_tokenize(sentence) #Word tokenization
        pos_tags = pos_tag(tokens)   #POS tagging
        chunks = ne_chunk(pos_tags, binary=True)
        for chunk in chunks:
            if type(chunk) == Tree and chunk.label() == 'NE':
                words, tags = zip(*chunk.leaves())
                all_tokens.append(' '.join(words))
            else:
                token = chunk[0]
                #                #Actually the stop words thing shouldn't matter, because these should get filtered out in tf-idf?
                #                #Do n-gramming here? Maybe just n-gramming instead of NER
                if token not in punctuation: # and token.lower() not in stopwords_eng:
                    all_tokens.append(token)

    normalizedTf = freqUtil.getNormalizedTermFrequency(all_tokens)

    #Lcase
    #These are unique terms so we can iterate over them
    for term in normalizedTf:
        if term not in idfs:
            idfs[term] = 1
        else:
            idfs[term] = idfs[term] + 1

    return normalizedTf
    #normalizedTf_desc = {k: v for k, v in sorted(normalizedTf.items(), key=lambda item: item[1], reverse=True)}
    #print(normalizedTf_desc)

    #This is biased - term frequency relies on occurrence counts, thus longer documents will be favoured more.
    #TO avoid this need to normalize the TF by diving each TF by the number of terms in the document



    #TODO: GET THE IDF FOR EACH DOCUMENT

docs = [ getFrequencyCounts(r"C:\Users\jacoo\Documents\Keywords\Corpus\Avengers_Assemble\Summaries.txt"),
         getFrequencyCounts(r"C:\Users\jacoo\Documents\Keywords\Corpus\Good_Will_Hunting\Summaries.txt"),
         getFrequencyCounts(r"C:\Users\jacoo\Documents\Keywords\Corpus\Les_Mis\Summaries.txt"),
         getFrequencyCounts(r"C:\Users\jacoo\Documents\Keywords\Corpus\Matrix\Summaries.txt"),
         getFrequencyCounts(r"C:\Users\jacoo\Documents\Keywords\Corpus\Shawshank_Redemption\Summaries.txt")]

numberOfDocuments = len(docs)

for doc in docs:
    for term in doc:
        #print(term + " --> " + str(doc[term]))
        tf = doc[term]
        idf = math.log(numberOfDocuments / idfs[term])
        doc[term] = tf * idf

for doc in docs:
    print("-----------------------------------------------------------------------------------")
    tfIdfSorted = {k: v for k, v in sorted(doc.items(), key=lambda item: item[1], reverse=True)}
    print(tfIdfSorted)


#TF
#           Doc1    Doc2    Doc3
# Will      1       2       1
# Life      2       3       1
# War       1       2       2


#
#if chunk not in stopwords_eng:
#    print(chunk)
#print(named_entities)
#break
#Named Entity Recognizer isn't very good - myb

#print(chunks)
#tokens=[word.lower() for word in words if word.isalpha()]
#for word in word_tokenize(sentence):
#    if word.isalpj
#if word not in stopwords_eng:
#    tokens.append(token)

#print(words)
#for word in word_tokenize(sentence):
#print(word_tokenize(sentence))

#Tag named entities or entities as a single item???
#NP chunking?
#N-grams
#Single grams
#Remove stop words

#print(sentences)
#print('\n-----\n'.join(sent_detector.tokenize(content.strip())))


#Remove stop words? NEs?
#print(content)

