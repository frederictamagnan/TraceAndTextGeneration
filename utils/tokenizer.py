import unicodedata
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from general_config import config


def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  # w = '<start> ' + w + ' <end>'
  return w

def preprocess_traces_dictionnary(traces_dictionnary):
    for client in traces_dictionnary.keys():
        for key in traces_dictionnary[client].keys():
            traces_dictionnary[client][key]=" ".join(traces_dictionnary[client][key])
    return traces_dictionnary

def tokenize_sequence(data, max_num_words, max_vocab_size,columns_to_select=["object", "action", "parameter", "return_code"]):
    """
    Tokenizes a given input sequence of words.
    Args:
        sentences: List of sentences
        filters: List of filters/punctuations to omit (for Keras tokenizer)
        max_num_words: Number of words to be considered in the fixed length sequence
        max_vocab_size: Number of most frequently occurring words to be kept in the vocabulary
    Returns:
        x : List of padded/truncated indices created from list of sentences
        word_index: dictionary storing the word-to-index correspondence
    """


    tensors=[]
    word_indexes=[]

    if config['experiment']=='traces':
        data=preprocess_traces_dictionnary(data)
        _word_tokenize_ref = lambda x: x.split(' ')
        _preprocess_sentence = lambda x:x

    elif config['experiment']=='text':
        columns_to_select=['text']
        _word_tokenize_ref = word_tokenize
        _preprocess_sentence = preprocess_sentence


    for column in columns_to_select:
        _word_tokenize=_word_tokenize_ref
        sentences=[data[key][column] for key in data.keys()]
        sentences = [' '.join(_word_tokenize(_preprocess_sentence(s))[:max_num_words]) for s in sentences]
        tokenizer = Tokenizer(filters="",split=" ",char_level=False)
        tokenizer.fit_on_texts(sentences)

        word_index = dict()
        word_index['PAD'] = 0
        word_index['UNK'] = 1
        word_index['GO'] = 2
        word_index['EOS'] = 3

        for i, word in enumerate(dict(tokenizer.word_index).keys()):
            word_index[word] = i + 4

        tokenizer.word_index = word_index
        x = tokenizer.texts_to_sequences(list(sentences))

        for i, seq in enumerate(x):
            if any(t >= max_vocab_size for t in seq):
                seq = [t if t < max_vocab_size else word_index['UNK'] for t in seq]
            seq.append(word_index['EOS'])
            x[i] = seq

        x = pad_sequences(x, padding='post', truncating='post', maxlen=max_num_words, value=word_index['PAD'])

        word_index = {k: v for k, v in word_index.items() if v < max_vocab_size}

        tensors.append(x)
        word_indexes.append(word_index)
    return tensors, word_indexes

if __name__=='__main__':

    sentences_text = {'sentence1':{"text": 'te quiero mucho mi amore de la vida loca de un pero malovamos a bailar'}}
    sentences_traces= {'client1': {'object': ['scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'scan0', 'caisse3', 'caisse3', 'caisse3', 'caisse3'], 'action': ['debloquer', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'scanner', 'transmission', 'abandon', 'ouvrirSession', 'ajouter', 'fermerSession', 'payer'], 'parameter': ['None', '3570590109324', '3017800238592', '3017620402678', '3245412567216', '3046920010856', '3017620402678', '8718309259938', '3560070048786', '8715700110622', '45496420598', '8718309259938', '3270190022534', '3560070048786', 'caisse3', 'None', 'None', '3570590109324', 'None', '0'], 'return_code': ['0', '-2', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '?', '0', '0', '0', '0']}}


    # tensors,word_indexes=tokenize_sequence(sentences_text,max_num_words=20,max_vocab_size=20)
    # print(x,word_index)
    tensors, word_indexes = tokenize_sequence(sentences_traces, max_num_words=20, max_vocab_size=20)
    print(tensors, word_indexes)