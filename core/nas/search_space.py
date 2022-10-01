import re
import numpy as np
import networkx as nx

class NASSearchSpace(object):
    ''' Defines the Search Space used to sample candidates by HiveNAS '''
         
    def __init__(self, config):
        '''
            Configurations are predefined in the Params class,
            the implementation should work given any set of operations' mapping
            and depth
        '''

        self.config = config
        # self.__initialize_graph()
                    

    def sample(self):
        '''
            Samples a random point in the search space
        '''

        # assert self.all_paths != None, 'Search space needs to be initialized!'

        # idx = np.random.randint(0, len(self.all_paths))
        # return self.__encode_path(self.all_paths[idx])

        path = ['input']

        for l in range(self.config['depth']):

            if self.config['allow_resnets'] and \
            np.random.rand() <= self.config['residual_blocks_rate']:
                sc_depth = np.random.randint(1, self.config['depth'] - l + 1)
                path.append('L{}_sc_{}'.format(l+1, sc_depth))

            path.append('L{}_{}'.format(l+1, np.random.choice(
                list(self.config['operations'].keys())
            )))
        
        path.append('output')

        return self.__encode_path(path)


    def get_neighbor(self, path_str):
        ''' Returns a path with 1-op difference (a neighbor)'''

        path = self.__strip_path(self.__decode_path(path_str))

        component = np.random.randint(1, len(path) - 1)

        ops = []
        if path[component].startswith('sc'):
            # modify skip-connection (either remove it or change residual depth)
            sc_max_depth = len([op for op in path[component:] if not op.startswith('sc')])
            ops = [f'sc_{i}' for i in range(sc_max_depth)]
            ops.remove(path[component])
        else:
            # modify operation
            ops = list(self.config['operations'].keys())
            ops.remove(path[component])
        
        # Replace randomly chosen component (operation) with any other op
        path[component] = np.random.choice(ops)

        # prune skip-connection if op == sc_0
        if path[component] == 'sc_0':
            del path[component]

        return self.__encode_path(path)


    def eval_format(self, path):
        ''' 
            Formats a path for evaluation (stripped, decoded, and
            excluding input/output layers) given a string-encoded path
        '''

        return self.__strip_path(self.__decode_path(path))[1:-1]


    def __initialize_graph(self):
        '''
            Initializes the search space DAG for easier sampling by the
            search algorithm
            [Deprecated] -- The search space DAG-representation is too memory-expensive
        '''
        
        self.dag = nx.DiGraph()
        self.dag.add_node('input')

        for l in range(self.config['depth']):
            for op in self.config['operations']:
                # Connect input layer to first hidden layer
                if l == 0:
                    self.dag.add_edges_from([('input', 
                                              'L{}_{}'.format(l+1, op))])
                    continue

                # Densely connect middle layers
                for prev_op in self.config['operations']:
                    self.dag.add_edges_from([('L{}_{}'.format(l, prev_op), 
                                              'L{}_{}'.format(l+1, op))])

                # Connect last hidden layer to output stem
                if l == self.config['depth'] - 1:
                    self.dag.add_edges_from([('L{}_{}'.format(l+1, op), 
                                              'output')])

        self.all_paths = list(nx.all_simple_paths(self.dag, 'input', 'output'))


    def __encode_path(self, path):
        ''' Returns a string encoding of a given path (list of ops)'''

        return '|'.join(self.__strip_path(path))


    def __decode_path(self, path):
        ''' Returns a list of ops given a string-encoded path '''

        ops = path.split('|')

        for i in range(1, len(ops) - 1):
            ops[i] = 'L{}_{}'.format(i, ops[i])

        return ops


    def __strip_path(self, path):
        ''' Strips path of layer ID prefixes given a list of ops '''
        
        return [re.sub('L\d+_', '', s) for s in path]


    def compute_space_size(self):
        ''' 
            Returns the number of possible architectures in the given space
            (i.e operations and depth) for analytical purposes
        '''

        return len(list(self.config['operations'].keys())) ** \
        self.config['depth']

        