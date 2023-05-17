"""Custom Commmand Line Argument Parser
"""

from argparse import ArgumentParser
from functools import partial

class ArgParser:

	@staticmethod
	def get_arguments(param_items):
		'''Parses the command line arguments

		Returns:
			dict: parsed arguments
		'''
		# parse arguments
		parser = ArgumentParser()

		parser.add_argument('-ea', '--evaluate-arch',
							type=bool,
							help='Manually evaluate an architecture (string-encoded)',
							default=None)
		parser.add_argument('-ft', '--fully-train',
							type=bool,
							help='Specifies whether to fully-train the best \
							candidate or perform the initial shallow NAS',
							choices=[True, False],
							default=False)
		parser.add_argument('-vb', '--verbose',
							help='Specifies whether to log all evaluation details',
							default=False,
							action='store_true')
		parser.add_argument('-c', '--config-file',
							type=str,
							help='Configuration file (relative) path',
							default=None)

		abbrevs = []
		for key, val in param_items:
			# TODO: consider adding :code:`help` argument to generated args list (parse from .py docstrings)
			if not isinstance(val, list) and not isinstance(val, dict) and not isinstance(val, partial):
				split_ls = key.lower().split('_')
				abbrev = '-' + ''.join([w[0] for w in split_ls])
				param = '--' + '-'.join(split_ls)

				if abbrev in abbrevs:
					# handle abbreviation conflicts
					abbrev = f'-{split_ls[0][0]}{split_ls[0][int(len(split_ls[0])/2)+1]}'

				parser.add_argument(abbrev, param, default=val, type=type(val))
				abbrevs.append(abbrev)


		args = parser.parse_args()
		
		return vars(args)

