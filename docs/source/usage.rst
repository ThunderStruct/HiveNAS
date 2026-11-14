.. _usage:

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

Usage
=====

Run these examples from the ``src/`` directory after installation.

Python API
----------

Select PSO through its optimizer class:

.. code-block:: python

   from config import Params
   from core import NASInterface, ParticleSwarmOptimization

   Params.set_parameter('ITERATIONS_COUNT', 20)
   Params.set_parameter('USE_NASBENCH', False)
   optimizer = ParticleSwarmOptimization(NASInterface(), swarm_size=10)
   results = optimizer.optimize()

For ABC or ACO, construct the corresponding optimizer and call ``optimize()``:

.. code-block:: python

   from core import ArtificialBeeColony, AntColonyOptimization

   optimizer = ArtificialBeeColony(NASInterface(), colony_size=10)
   # Or:
   optimizer = AntColonyOptimization(NASInterface(), colony_size=10)

For NASBench-101 evaluation, set ``USE_NASBENCH`` to ``True`` before
constructing ``NASInterface()``. See the installation instructions for the
API and dataset setup.

NASBench-101 scores
-------------------

The adapter uses mean test accuracy at ``NASBENCH_BUDGET`` as search fitness.
The ``final_acc`` result records mean 108-epoch test accuracy when available;
otherwise it uses the search-budget score. These are benchmark search scores,
not an independent held-out evaluation.

ABC command line
----------------

The ``HiveNAS.py`` entry point runs ABC. Use the Python API above for PSO
and ACO.

.. code-block:: bash

   python HiveNAS.py --verbose --iterations-count 20
   python HiveNAS.py --verbose --config-file path/to/config.yaml

Advanced Usage
--------------

To run HiveNAS over the `MNIST <https://www.tensorflow.org/datasets/catalog/mnist>`_ benchmark, specify a unique configuration version name and the dataset:

.. code-block:: bash

  python HiveNAS.py --verbose -cv="mnist-trial" -oo="NAS" -ds="MNIST"


The same setting over the `FashionMNIST <https://www.tensorflow.org/datasets/catalog/fashion_mnist>`_ benchmark can be run as follows:

.. code-block:: bash

  python HiveNAS.py --verbose -cv="fashion-mnist-trial" -oo="NAS" -ds="FASHION_MNIST"


To experiment with Artificial Bee Colony and Numerical Benchmarks (:class:`Rosenbrock optimization <benchmarks.rosenbrock>`, for instance), the :code:`-oo` flag (or :code:`--optimization-objective`) can be specified.

.. code-block:: bash

  python HiveNAS.py --verbose -cv="rosenbrock-trial" -oo="Rosenbrock"


CLI Arguments
~~~~~~~~~~~~~

To override any of the :ref:`default parameters <parameters-table>`, refer to the table below (or use the :bash:`-h` / :bash:`--help` argument):

.. rst-class:: arguments-cli-table

.. table::
  :widths: 25 45 30

  ==================================  ==========================================  ===============
       Configuration Parameter                       Argument Name                 Argument Flag
  ==================================  ==========================================  ===============
    CONFIG_VERSION                     :bash:`--config-version`                    :bash:`-cv`
    SEED_VALUE                         :bash:`--seed-value`                        :bash:`-sv`
    OPTIMIZATION_OBJECTIVE             :bash:`--optimization-objective`            :bash:`-oo`
    ABANDONMENT_LIMIT                  :bash:`--abandonment-limit`                 :bash:`-al`
    COLONY_SIZE                        :bash:`--colony-size`                       :bash:`-cs`
    EMPLOYEE_ONLOOKER_RATIO            :bash:`--employee-onlooker-ratio`           :bash:`-eor`
    ITERATIONS_COUNT                   :bash:`--iterations-count`                  :bash:`-ic`
    RESULTS_SAVE_FREQUENCY             :bash:`--results-save-frequency`            :bash:`-rsf`
    RESULTS_BASE_PATH                  :bash:`--results-base-path`                 :bash:`-rbp`
    HISTORY_FILES_SUBPATH              :bash:`--history-files-subpath`             :bash:`-hfs`
    ENABLE_WEIGHT_SAVING               :bash:`--enable-weight-saving`              :bash:`-ews`
    WEIGHT_FILES_SUBPATH               :bash:`--weight-files-subpath`              :bash:`-wfs`
    RESUME_FROM_RESULTS_FILE           :bash:`--resume-from-results-file`          :bash:`-rfrf`
    DEPTH                              :bash:`--depth`                             :bash:`-d`
    STOCHASTIC_SC_RATE                 :bash:`--stochastic-sc-rate`                :bash:`-ssr`
    DATASET                            :bash:`--dataset`                           :bash:`-ds`
    EPOCHS                             :bash:`--epochs`                            :bash:`-e`
    MOMENTUM_EPOCHS                    :bash:`--momentum-epochs`                   :bash:`-me`
    FULL_TRAIN_EPOCHS                  :bash:`--full-train-epochs`                 :bash:`-fte`
    TERMINATION_THRESHOLD_FACTOR       :bash:`--termination-threshold-factor`      :bash:`-ttf`
    TERMINATION_DIMINISHING_FACTOR     :bash:`--termination-diminishing-factor`    :bash:`-tdf`
    INITIAL_LR                         :bash:`--initial-lr`                        :bash:`-il`
    FINAL_LR                           :bash:`--final-lr`                          :bash:`-fl`
    BATCH_SIZE                         :bash:`--batch-size`                        :bash:`-bs`
    AFFINE_TRANSFORMATIONS_ENABLED     :bash:`--affine-transformations-enabled`    :bash:`-ate`
    CUTOUT_PROB                        :bash:`--cutout-prob`                       :bash:`-cp`
    SATURATION_AUG_PROB                :bash:`--saturation-aug-prob`               :bash:`-sap`
    CONTRAST_AUG_PROB                  :bash:`--contrast-aug-prob`                 :bash:`-cap`
  ==================================  ==========================================  ===============


Alternative Datasets
~~~~~~~~~~~~~~~~~~~~

In addition to the predefined datasets, virtually any labeled type of data can be used by defining a custom loader in :class:`~core.nas.evaluation_strategy.NASEval`\'s initializer.

.. note::
  Ensure that the input data (i.e :python:`(X_train, X_test)`) have a 4-dimensional shape, matching `Keras' CIFAR10 <https://keras.io/api/datasets/cifar10/>`_ . If the data has fewer channels, consider adding placeholders as demonstrated in :class:`~core.nas.evaluation_strategy.NASEval` with MNIST/FashionMNIST (:python:`X_train.reshape(-1,28,28,1)`)

**HiveNAS** currently supports CNNs only, with plans to expand to RNNs in the future.
