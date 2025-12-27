"""User prompts' handle.
"""

import os

class PromptHandler:
    '''Wrapper for input prompt-handling methods
    '''

    @staticmethod
    def prompt_yes_no(question, default='y'):
        '''Yes/no query; reverts to default value if no input is given

        Args:
            question (str): printed prompt question
            default (str, optional): user answer to revert to if no \
            response is given (empty input) ; defaults to "yes"

        Returns:
            bool: user response
        '''

        # Auto-answer 'yes' if environment variable is set
        if os.environ.get('SWARMNAS_AUTO_OVERWRITE') == '1':
            print(f'{question} (y/n): y [auto]')
            return True

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

