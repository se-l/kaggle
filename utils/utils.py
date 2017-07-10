import logging

class dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

class Logger:
    logger = None

    @staticmethod
    def init_log(output=None):
        """
        Initialise the logger
        """
        Logger.logger = logging.getLogger('patternFinder')
        # Logger.logger.setLevel(logging.ERROR)
        # Logger.logger.setLevel(logging.INFO)
        Logger.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s')
        # if output is not None:
        slogger = logging.StreamHandler()
        slogger.setFormatter(formatter)
        Logger.logger.addHandler(slogger)
        # else:
        if output is not None:
            flogger = logging.FileHandler(output)
            flogger.setFormatter(formatter)
            Logger.logger.addHandler(flogger)

    # @staticmethod
    # def info(method, str):
    #     """
    #     Write info log
    #     :param method: Method name
    #     :param str: Log message
    #     """
    #     Logger.logger.info('[%s]\n%s\n' % (method, str))
    #
    # @staticmethod
    # def error(method, str):
    #     """
    #     Write info log
    #     :param method: Method name
    #     :param str: Log message
    #     """
    #     Logger.logger.error('[%s]\n%s\n' % (method, str))

    @staticmethod
    def info(str):
        """
        Write info log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.info('%s\n' % (str))

    @staticmethod
    def debug(str):
        """
        Write info log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.debug('%s\n' % (str))

    @staticmethod
    def error(str):
        """
        Write info log
        :param method: Method name
        :param str: Log message
        """
        Logger.logger.error('%s\n' % (str))
