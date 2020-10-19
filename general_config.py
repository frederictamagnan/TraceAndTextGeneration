import logging
import os
from pathlib import Path
import inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

general_config = dict(
    # General Parameters
    experiment='text',
    data_dir='C:/Users/QZTD9928/Documents/code/DeepLearningOnTracesVsText/data/',
    raw_traces_data_name="100043-steps.csv",
    raw_text_data_name='snli_1.0_train.jsonl',
    traces_data_name='scannette.json',
    text_data_name='snli_train.json',

    logging_level=logging.DEBUG,

    #hyperparameters preprocessing
    max_num_words=20,
    max_vocab_size=80,
    batch_size=20,

)

def Hey():
    pass

root=str(Path(inspect.getfile(Hey)).parent)
general_config['root']=root