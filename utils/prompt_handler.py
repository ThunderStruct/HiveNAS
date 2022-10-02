
class PromptHandler:
    '''
        Wrapper for input prompt-handling methods
    '''


    @staticmethod
    def prompt_yes_no(question, default='y'):
        '''
            Yes/no query; reverts to default value if no input is given
        '''

        valid_res = {
            'yes': True,
            'y': True,
            'no': False,
            'n': False
        }

        choice = None

        while choice not in valid_res:
            choice = input(f'{question} (y/n): ').lower().replace(' ', '') or default

        return valid_res[choice]

