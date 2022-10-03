import time

class Logger:
    '''
        Wrapper for debug- and info-logging methods
    '''

    __DEBUG_PREFIX  = 'DEBUG:'
    __STATUS_PREFIX = 'STATUS:'
    __EVAL_PREFIX = 'EVALUATION LOG:'
    __FILESAVE_PREFIX = 'FILE-SAVE SUCCESSFUL:'

    __START_TIME = None

    EVALUATION_LOGGING = False


    @staticmethod
    def debug(msg=None):
        ''' Debugging messages '''

        print('{} {}'.format(Logger.__DEBUG_PREFIX,
                            ('MARK' if msg is None else str(msg))))

    
    @staticmethod
    def status(itr, msg=None):
        ''' Generic logging '''

        print('{} itr: {} -- {}'.format(Logger.__STATUS_PREFIX,
                                        str(itr),
                                        ('MARK' if msg is None else str(msg))))


    @staticmethod
    def evaluation_log(type, id, candidate_pos):
        ''' Logs pre-evaluation info for every candidate '''

        if not Logger.EVALUATION_LOGGING:
            return

        print('\n{} {} ID ({}) -- Candidate ({})\n'.format(Logger.__EVAL_PREFIX,
                                                           type,
                                                           str(id),
                                                           str(candidate_pos)))


    @staticmethod
    def filesave_log(candidate, filename):
        ''' Logs candidate info upon file-save '''

        if not Logger.EVALUATION_LOGGING:
            return

        print('\n{} Candidate ({}) was saved to {}\n'.format(Logger.__FILESAVE_PREFIX,
                                                             str(candidate),
                                                             filename))
        

    @staticmethod
    def start_log():
        ''' Logs the start msg and intializes the global timer '''

        Logger.__START_TIME = time.time()
        dashes = '------------------------'
        print('{}\n-- OPTIMIZATION START --\n{}'.format(dashes, dashes))


    @staticmethod
    def end_log():
        ''' Logs total time taken upon optimization end '''

        end_time = time.time() - Logger.__START_TIME
        dashes = '---------------------'
        print('{}\n-- OPTIMIZATION END --\n{} === TOTAL TIME TAKEN: {} ==== \n'.format(dashes, dashes, end_time))

