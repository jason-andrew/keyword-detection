








from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser




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

#Obtained with
#- import nltk
#- nltk.download('tagsets')
#- nltk.help.upenn_tagset()
nltk_punctuation_tags = ["$", "''", "(", ")", ",", "--", ".", ":", "SYM"]

#grammar = "NP: {<DT>?<JJ>*<NN>?} {<NNP>+}"\
#CHUNK SEQUENCES
grammar = r"""
  NP: {<NN><IN><NN|NNS>}                # chunk sequences of proper nouns ('gift for mathematics')
      {<NNP>+}     #Proper nouns (singular)
      {<NNPS>+}    #Proper nouns (plural)
      {<NN>+}      #Common nouns (singular or mass)
      {<NNS>+}     #Common nouns (plural)
      {<PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
       
    """
cp = nltk.RegexpParser(grammar)

#Take all the keywords,

#COuld learn those patterns from text. Grab keywords from DB and look at plots for the keywords, then identify the POS

def getNgrams(tokens, n):
    ngrams = []
    for i in range(0, len(tokens)-n+1):
        ngrams.append(' '.join(tokens[i: i+n]))
    print(ngrams)
    return ngrams

def getWord(tuple):
    return tuple[0]

def getPOS(tuple):
    return tuple[1]

def get_continuous_chunks(text, chunk_func=ne_chunk):
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def getFrequencyCounts(filename):
    content = fileUtil.readData(filename)

    #print(nltk.help.upenn_tagset())
    all_tokens = []

    sentences = sent_detector.tokenize(content.strip())
    for sentence in sentences:
        #nes = get_continuous_chunks(sentence)

        #Tokenize and POS tag
        chunks = cp.parse(pos_tag(word_tokenize(sentence)))
        for chunk in chunks:
            if type(chunk) == Tree:
                words, tags = zip(*chunk.leaves())
                all_tokens.append(' '.join(words).lower())

    normalizedTf = freqUtil.getNormalizedTermFrequency(all_tokens)

#Lcase
#These are unique terms so we can iterate over them
    for term in normalizedTf:
        if term not in idfs:
            idfs[term] = 1
        else:
            idfs[term] = idfs[term] + 1

    return normalizedTf

#  return normalizedTf
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
         #]
numberOfDocuments = len(docs)

#nltk.download('tagsets')

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