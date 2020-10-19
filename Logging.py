import logging

from logging.handlers import RotatingFileHandler
from general_config import general_config
class Logging:
    def __init__(self):

        # création de l'objet logger qui va nous servir à écrire dans les logs
        self.logger = logging.getLogger()
        # on met le niveau du logger à DEBUG, comme ça il écrit tout
        self.logger.setLevel(general_config['logging_level'])

        # création d'un formateur qui va ajouter le temps, le niveau
        # de chaque message quand on écrira un message dans le log
        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
        # création d'un handler qui va rediriger une écriture du log vers
        # un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
        # root='C:/Users/QZTD9928/Documents/code/DeepLearningOnTracesVsText/'
        root=general_config['root']
        file_handler = RotatingFileHandler(root+'/'+'activity.log', 'a', 1000000, 1)
        # on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
        # créé précédement et on ajoute ce handler au logger
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # création d'un second handler qui va rediriger chaque écriture de log
        # sur la console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)

        # Après 3 heures, on peut enfin logguer
        # Il est temps de spammer votre code avec des logs partout :

    @staticmethod
    def format(text):
        char = ' --- '
        return char+text+char

    def info(self,text):
        self.logger.info(Logging.format(text))
    def warning(self,text):
        warning='/!\\'
        self.logger.warning(Logging.format(warning+text+warning))
    def debug(self,text):
        self.logger.debug(Logging.format(text))