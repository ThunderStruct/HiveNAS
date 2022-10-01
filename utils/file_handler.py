import os
import pickle
import pandas as pd


class FileHandler:
    '''
        Wrapper for file-handling methods
    '''
    
    __VALID_PATHS = {}

    @staticmethod
    def __path_exists(path):
        ''' Checks if file exists '''

        return os.path.exists(path)


    @staticmethod
    def validate_path(path, ignore_abs_format=True):
        ''' 
            Ensures that a given path is universaly valid
            (Windows/Linux/MacOS/POSIX)
        '''

        # Directory already exists, prompt for overwrite permission (first time only)
        if path not in FileHandler.__VALID_PATHS and FileHandler.__path_exists(path):

            FileHandler.__VALID_PATHS[path] = True
            
            if len(os.listdir(path)) == 0:
                # directory exists and is empty -> is valid
                return FileHandler.__VALID_PATHS[path]

            # directory exists and is NOT empty
            FileHandler.__VALID_PATHS[path] = True
            print(f'\nPath ({path}) already exists!\n\n')
            return PromptHandler.prompt_yes_no('Would you like to overwrite files in this path?')

        # Previously evaluated
        if path in FileHandler.__VALID_PATHS:
            return FileHandler.__VALID_PATHS[path]
        
        # Check the validity of the given path
        try:
            os.makedirs(path)
            FileHandler.__VALID_PATHS[path] = True
        except OSError as e:
            # path invalid
            print('\nBase path and/or config version are invalid! Please choose path-friendly names.\n')
            FileHandler.__VALID_PATHS[path] = False

        return FileHandler.__VALID_PATHS[path]


    @staticmethod
    def create_dir(path):
        ''' Recursively creates new directory if it does not exist '''

        if not FileHandler.__path_exists(path):
            os.makedirs(path)


    @staticmethod
    def path_must_exist(path):
        ''' 
            Checks if file exists and raises error if it does not.
            Used when the logic of the algorithm depends on the loaded file
        '''

        if not FileHandler.__path_exists(path):
            raise FileNotFoundError(errno.ENOENT, 
                                    os.strerror(errno.ENOENT), 
                                    path)


    @staticmethod
    def save_pickle(p_dict, path, filename, force_dir=True):
        ''' Saves the given dictionary as pickle '''

        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                # directory does not exist and cannot create dir
                return False
        
        # dump pickle
        with open(os.path.join(path, filename), 'wb') as handle:
            pickle.dump(p_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return True


    @staticmethod
    def load_pickle(path, default_dict={}):
        ''' Loads pickle and returns decoded dictionary '''

        res = default_dict

        if FileHandler.__path_exists(path):
            with open(path, 'rb') as handle:
                res = pickle.load(handle)

        return res


    @staticmethod
    def save_df(df, path, filename, force_dir=True):
        ''' Saves the given dataframe to a given path+filename '''

        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                # directory does not exist and cannot create dir
                return False

        # save dataframe
        df.to_csv(os.path.join(path, filename))


    @staticmethod
    def load_df(path, default_df=None):
        ''' Loads dataframe if it exists or defaults to an empty df '''

        res = default_df or pd.DataFrame()

        if FileHandler.__path_exists(path):
            res = pd.read_csv(path, header=0, index_col=0)

        return res

    
    @staticmethod
    def export_yaml(config_dict, path, filename, file_version_comment='', force_dir=True):
        ''' Exports a given dictionary to a yaml file '''
        
        if not FileHandler.__path_exists(path):
            if force_dir:
                FileHandler.create_dir(path)
            else:
                return False
            
        with open(os.path.join(path, filename), 'w') as handler:
            if file_version_comment:
                handler.write(f'\n# {file_version_comment}\n\n')
            yaml.dump(config_dict, handler, default_flow_style=False)

        return True

    
    @staticmethod
    def load_yaml(path, loader, default_dict={}):
        ''' Loads a yaml config file and returns it as dict '''

        res = default_dict

        if FileHandler.__path_exists(path):
            with open(path, 'r') as handler:    
                res = yaml.load(handler, Loader=loader)

        return res

