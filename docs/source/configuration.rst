.. _configuration:

.. role:: python(code)
   :language: python

.. role:: yaml(code)
   :language: yaml

.. role:: param_default
   :class: param_default

Configuration
=============

.. raw:: html
   
   <br />

Framework Customization
-----------------------

HiveNAS has a flexible and configurable design to fit a wider range of task contexts. Below is the list of all operational parameters, their descriptions, and default values.

Custom `YAML <https://yaml.org/>`_ tags are defined within the framework to keep track of configurations, as well as provide a readable and user-friendly framework operation.

.. raw:: html
   
   <br />


Allowed YAML Tags
-----------------

Configuration files accept two custom YAML tags: :yaml:`!Operation` and :yaml:`!!python/tuple [n, m]` (following the `YAML 2002 specs <https://yaml.org/spec/history/2002-04-07.html#trans-seq>`_) to represent :math:`n \times m` shapes. The definition of the :python:`constructor` and :python:`representer` for each tag is defined in :func:`config.params.Params.init_from_yaml` and :func:`config.params.Params.export_yaml`, respectively.

.. raw:: html

   <br />

.. _parameters-table:

Parameters Table
-----------------


In addition to the documentation below, the `default configuration file <https://github.com/ThunderStruct/HiveNAS/blob/main/src/config/settings/config_default.yaml>`_ is thoroughly documented and contains all the default values used by the framework.


.. raw:: html
   
   <br />

.. rst-class:: config-table

==================================  ================  ==============================================
 Parameter Key                         Data Type                  Parameter Description 
==================================  ================  ==============================================
  *CONFIG_VERSION*                    :python:`str`      | The simulation's configuration version. \
                                                           All generated results will be under this configuration name
                                                         | *Note: must be a file-path compatible string*.
                                                         |
                                                         | :param_default:`Default:` :python:`'default_config'`

  *OPTIMIZATION_OBJECTIVE*            :python:`str`      | The objective to be optimized.
                                                         | *Valid options:* :python:`['NAS', 'Rosenbrock', 'Sphere_min', 'Sphere_max']`.
                                                         |
                                                         | :param_default:`Default:` :python:`'NAS'`

  *ABANDONMENT_LIMIT*                 :python:`int`      | The ABC optimizer abandonment limit.
                                                         |
                                                         | :param_default:`Default:` :python:`3`

  *COLONY_SIZE*                       :python:`int`      | Number of Employee and Onlooker bees in the colony. \
                                                           Scouts have a 1-to-1 ratio with Employees *(as per the classical ABC implementation)*.
                                                         |
                                                         | :param_default:`Default:` :python:`7`

  *EMPLOYEE_ONLOOKER_RATIO*          :python:`float`     | Employees to Onlookers ratio 
                                                         | *i.e* :math:`Employees = Colony * Ratio`
                                                         |
                                                         | :param_default:`Default:` :python:`0.44`

  *ITERATIONS_COUNT*                  :python:`int`      | Number of ABC iteractions
                                                         | *Note: this is not the number of training epochs per \ candidate*.
                                                         |
                                                         | :param_default:`Default:` :python:`15`

  *RESULTS_SAVE_FREQUENCY*            :python:`int`      | Update the main CSV data file every :math:`n` evaluations 
                                                         | *Note: (evaluations not iterations i.e* :code:`ITERATIONS_COUNT` :math:`\times` *(* :code:`COLONY_SIZE` :math:`/` :code:`RESULTS_SAVE_FREQUENCY` *)* :math:`=` *total saves)*.
                                                         |
                                                         | :param_default:`Default:` :python:`1`

  *RESULTS_BASE_PATH*                 :python:`str`      | Base directory path, \
                                                           where a :code:`CONFIG_VERSION`-named folder is created, storing all generated results.
                                                         | *The path gets recursively created if it does not exist.*
                                                         |
                                                         | :param_default:`Default:` :python:`'./results/'`

  *HISTORY_FILES_SUBPATH*             :python:`str`      | Relative sub-path for training history files folder.
                                                         | *The path gets recursively created if it does not exist.*
                                                         |
                                                         | :param_default:`Default:` :python:`'training_history/'`

  *ENABLE_WEIGHT_SAVING*              :python:`bool`     | Specifies whether or not to save candidate model files.
                                                         | *Note: each model file may take up a large amount of disk space (1-2gb on average). Ensure that enough disk space is available to avoid optimization interruptions*.
                                                         |
                                                         | :param_default:`Default:` :python:`False`

  *WEIGHT_FILES_SUBPATH*              :python:`str`      | Relative sub-path for candidate model files folder\
                                                           (if :code:`ENABLE_WEIGHT_SAVING` is set to :python:`True`)
                                                         | *The path gets recursively created if it does not exist.*
                                                         |
                                                         | :param_default:`Default:` :python:`'weights/'`

  *RESUME_FROM_RESULTS_FILE*          :python:`bool`     | Specifies whether or not to resume training from the \
                                                           main data file (if it exists).
                                                         | *Note: this might affect ABC's convergence behavior as some internal optimizer settings will be set to default*
                                                         |
                                                         | :param_default:`Default:` :python:`False`

  *DEPTH*                             :python:`int`      | The candidate models' fixed depth (excluding \
                                                           :code:`INPUT_STEM` and :code:`OUTPUT_STEM`)
                                                         |
                                                         | :param_default:`Default:` :python:`4`

  *OPERATIONS*                        :python:`dict`     | A dictionary defining a :code:`search_space` \
                                                           :python:`list` (searchable operations) and a :code:`reference_space` :python:`dict` (a lookup table for discretized operations/hyperparameters). These \
                                                           along with the :code:`DEPTH` define the Search Space, and therefore largely influence the performance of the framework.
                                                         |
                                                         | The :code:`search_space` / :code:`reference_space` allow hybrid layer-wise and cell-based operations (by defining a cell in :class:`~config.operation_cells.OperationCells` and treating it as any Keras layer (see built-in example cells)).
                                                         |
                                                         | *Note: operations in the :code:`reference_space` are defined as partial functions* \
                                                           (:python:`functools.partial`) *and can be specified in YAML format using the custom tag* :yaml:`!Operation` *(* `see example config file <https://github.com/ThunderStruct/HiveNAS/blob/main/src/config/settings/config_default.yaml>`_ *)*.
                                                         |
                                                         | :param_default:`Default:` *(refer to example file)*

  *STOCHASTIC_SC_RATE*               :python:`float`     | Rate at which skip-connections could occur \ 
                                                           per layer. The depth of the residual block is randomly sampled (bounded between [1, :code:`DEPTH` - *current_layer*])
                                                         | A value of :python:`0.0` disables ResNets, \ 
                                                           while a value of :python:`1.0` guarantees a skip-connection between all operations.
                                                         |
                                                         | :param_default:`Default:` :python:`0.0`

  *DATASET*                           :python:`str`      | The optimization problem's dataset (for \
                                                           :python:`OPTIMIZATION_OBJECTIVE = 'NAS'`).
                                                         | *Valid options:* :python:`['CIFAR10', 'MNIST', 'FASHION_MNIST']` *. The framework is dataset-agnostic and should function with any other dataset, provided that its loader is defined in* :class:`~core.nas.evaluation_strategy.NASEval` *.*
                                                         |
                                                         | :param_default:`Default:` :python:`'CIFAR10'`

  *INPUT_STEM*                        :python:`list`     | A list of operations' keys defining the static input \
                                                           stem for all candidates.
                                                         | *Note: operations referenced here must be defined in :code:`OPERATIONS.reference_space`* \
                                                           (:python:`functools.partial`) *and can be specified in YAML format using the custom tag* :yaml:`!Operation` *(* `see example config file <https://github.com/ThunderStruct/HiveNAS/blob/main/src/config/settings/config_default.yaml>`_ *)*.
                                                         |
                                                         | :param_default:`Default:` *(refer to example file)*

  *OUTPUT_STEM*                       :python:`list`     | A list of operations defining the static output \
                                                           stem for all candidates.
                                                         | *Note: operations referenced here must be defined in :code:`OPERATIONS.reference_space`* \
                                                           (:python:`functools.partial`) *and can be specified in YAML format using the custom tag* :yaml:`!Operation` *(* `see example config file <https://github.com/ThunderStruct/HiveNAS/blob/main/src/config/settings/config_default.yaml>`_ *)*.
                                                         |
                                                         | :param_default:`Default:` *(refer to example file)*

  *EPOCHS*                            :python:`int`      | Number of training epochs per candidate.
                                                         | *Note: it is empirically deduced that any number above* \
                                                           :python:`10` *significantly impacts the NAS convergence process and limits the exploration/exploitation of ABC. A shallow initial search provides a sufficiently good measure of a candidate's performance.*
                                                         |
                                                         | :param_default:`Default:` :python:`5`

  *FULL_TRAIN_EPOCHS*                 :python:`int`      | Number of training epochs to train \ 
                                                           the best-performing candidate resulting from the shallow search (used by :func:`~core.nas.nas_interface.NASInterface.fully_train_best_model`).
                                                         |
                                                         | :param_default:`Default:` :python:`100`

  *LR*                               :python:`float`     | The learning rate used when evaluating candidates.
                                                         |
                                                         | *Note: this parameter overrides the :code:`learning_rate` defined in the :code:`OPTIMIZER` partial. A value of :python:`0.0` disables it*
                                                         | :param_default:`Default:` :python:`0.0`

  *BATCH_SIZE*                        :python:`int`      | The candidates' training batch size.
                                                         |
                                                         | :param_default:`Default:` :python:`128`

  *OPTIMIZER*                         :python:`str`      | The Evaluation Strategy's optimizer, defined \
                                                           as partial functions (:python:`functools.partial`)
                                                         | *Included optimizers:* :python:`Adam`, :python:`SGD`, :python:`RMSprop`.
                                                         |
                                                         | *Note: define custom optimizers by simply importing them to* :class:`~config.params.Params` *(must be availble in the* :python:`globals()` *variable).*
                                                         |
                                                         | :param_default:`Default:` :python:`partial(SGD, learning_rate=0.08, decay=5e-4, momentum=0.9, nesterov=True)`    

  *AFFINE_TRANSFORMATIONS_ENABLE*     :python:`bool`     | Enables simple affine transformations \
                                                           *(rotation, shift, zoom, sheer, flip -- customize in the* :class:`~core.nas.evaluation_strategy.NASEval` *class).*
                                                         |
                                                         | :param_default:`Default:` :python:`True`

  *CUTOUT_PROB*                      :python:`float`     | Probability of applying cutout augmentation per sample.
                                                         | A value of :python:`0.0` disables cutout augmentation, while \
                                                           a value of :python:`1.0` guarantees the augmentation for every sample.
                                                         |
                                                         | :param_default:`Default:` :python:`0.5`

  *SATURATION_AUG_PROB*              :python:`float`     | Probability of applying saturation augmentation per sample.
                                                         | A value of :python:`0.0` disables saturation augmentation, while \
                                                           a value of :python:`1.0` guarantees the augmentation for every sample.
                                                         |
                                                         | :param_default:`Default:` :python:`0.75`    

  *CONTRAST_AUG_PROB*                :python:`float`     | Probability of applying contrast augmentation per sample.
                                                         | A value of :python:`0.0` disables contrast augmentation, while \
                                                           a value of :python:`1.0` guarantees the augmentation for every sample.
                                                         |
                                                         | :param_default:`Default:` :python:`0.75`

  *MOMENTUM_EPOCH*                    :python:`int`      | The number of epochs in the *Momentum Evaluation* pool to \
                                                           be assigned to candidates with a stable convergence profile.
                                                         | A value of :python:`0` disables Momentum Evaluation.
                                                         |
                                                         | :param_default:`Default:` :python:`0`

  *TERMINATION_THRESHOLD_FACTOR*     :python:`float`     | Threshold factor (:math:`β`) for *ACT* \
                                                           (:class:`~core.nas.act.TerminateOnThreshold`).
                                                         | A value of :python:`0.0` disables ACT
                                                         |
                                                         | :param_default:`Default:` :python:`0.25`

  *TERMINATION_DIMINISHING_FACTOR*   :python:`float`     | Diminishing factor (:math:`ζ`) for *ACT* \
                                                           (:class:`~core.nas.act.TerminateOnThreshold`).
                                                         |
                                                         | :param_default:`Default:` :python:`0.25`

==================================  ================  ==============================================