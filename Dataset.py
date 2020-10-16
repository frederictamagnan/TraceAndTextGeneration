
from utils.tokenizer import tokenize_sequence
from general_config import general_config
import tensorflow as tf
from Logging import Logging
from pathlib import Path,PurePath
import json
class Dataset:

    def __init__(self,limit_rows=False):
        assert tf.executing_eagerly()
        self.log=Logging()

        self.log.info("Program starts")
        tensors,self.word_indexes=self.load_data(limit_rows=limit_rows)

        self.log.info("Data has been tokenized")
        self.log.info(str(len(tensors[0]))+" rows in total")
        self.channels = len(tensors)
        assert (config['experiment']=='traces' or (config['experiment']=='text' and self.channels == 1))

        datasets=[]
        for tensor in tensors:
            datasets.append(tf.data.Dataset.from_tensor_slices(tensor))

        self.dataset=tf.data.Dataset.zip(tuple(datasets))

        self.batched_dataset = self.dataset.batch(config['batch_size'])
        self.log.info("Batched dataset created")

    def load_data(self,limit_rows=False):
        if config['experiment']=='text':
            name_csv = config['text_data_name']
            word='sentence'
        elif config['experiment']=='traces':
            name_csv= config['traces_data_name']
            word='client'

        dir = PurePath(config['data_dir'])
        file_name = Path.joinpath(dir, name_csv)
        with open(file_name, 'r') as f:
            _dict = json.load(f)
        if limit_rows:
            for i in range(limit_rows, len(_dict.keys())):
                del _dict[word + str(i)]
        self.channel_names=_dict[word+str(0)].keys()
        self.log.info("Loaded data as dict")
        tensors, words_indexes = tokenize_sequence(_dict, max_num_words=config['max_num_words'],

                                                   max_vocab_size=config['max_vocab_size'])

        return tensors,words_indexes

if __name__=='__main__':
    dataset=Dataset()
