import logging as lg

logger = lg.getLogger(__name__)
logger.setLevel(lg.INFO)

formatter = lg.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

file_handler = lg.FileHandler('log.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

class logg:


    def debug(self,debug):
        try:

            logger.debug(debug)
        except Exception as e:
            logger.exception(e+"During logging")

    def info(self,info):
        try:
            logger.info(info)
        except Exception as e:
            logger.exception(e+"During logging")

    def warning(self,warning):
        try:
            logger.warning(warning)
        except Exception as e:
            logger.exception(e+"During logging")

    def error(self,error):
        try:
            logger.error(error)
        except Exception as e:
            logger.exception(e+"During logging")

    def critical(self,critical):
        try:
            logger.critical(critical)
        except Exception as e:
            logger.exception(e+"During logging")

    def excpt(self,exception):
        try:
            logger.exception(exception)
        except Exception as e:
            logger.exception(e+"During logging")