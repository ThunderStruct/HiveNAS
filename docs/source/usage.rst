.. _usage:

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

Usage
=====

The top-level entry point of the framework is the :class:`~HiveNAS` class.

Examples
--------

The simplest way to run **HiveNAS** is to use the default parameters (which are empirically optimized for simple image-classification tasks) as follows:

.. code-block:: bash

  python HiveNAS.py --verbose


To specify a custom configuration file (see :ref:`Configuration <configuration>`), use the :code:`-c | --config-file` argument:

.. code-block:: bash

  python HiveNAS.py --verbose -c="path/to/config.yaml"


All primitive, non-iterable :ref:`operational parameters <parameters-table>` (i.e :python:`int`, :python:`float`, :python:`bool`, :python:`str`) are defined as CLI arguments. Find more customizable use cases below.


Advanced Usage
--------------

To run HiveNAS over the `MNIST <https://www.tensorflow.org/datasets/catalog/mnist>`_ benchmark, specify a unique configuration version name and the dataset:

.. code-block:: bash

  python HiveNAS.py --verbose -cv="mnist-trial" -oo="NAS" -ds="MNIST"


The same setting over the `FashionMNIST <https://www.tensorflow.org/datasets/catalog/fashion_mnist>`_ benchmark can be ran as follows:

.. code-block:: bash

  python HiveNAS.py --verbose -cv="fashion-mnist-trial" -oo="NAS" -ds="FASHION_MNIST"


To experiment with Artificial Bee Colony and Numerical Benchmarks (:class:`Rosenbrock optimization <benchmarks.rosenbrock>`, for instance), the :code:`-oo` flag (or :code:`--optimization-objective`) can be specified.

.. code-block:: bash

  python HiveNAS.py --verbose -cv="rosenbrock-trial" -oo="Rosenbrock"


CLI Arguments
~~~~~~~~~~~~~

To override any of the :ref:`default parameters <parameters-table>`, refer to the table below:

.. rst-class:: arguments-cli-table

.. table::
  :widths: 25 45 30

  ==================================  ==========================================  ===============
       Configuration Parameter                       Argument Name                 Argument Flag
  ==================================  ==========================================  ===============
    CONFIG_VERSION                     :bash:`--config-version`                    :bash:`-cv`
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
    RESIDUAL_BLOCKS_RATE               :bash:`--residual-blocks-rate`              :bash:`-rbr`
    DATASET                            :bash:`--dataset`                           :bash:`-ds`
    EPOCHS                             :bash:`--epochs`                            :bash:`-e`
    MOMENTUM_EPOCHS                    :bash:`--momentum-epochs`                   :bash:`-me`
    FULL_TRAIN_EPOCHS                  :bash:`--full-train-epochs`                 :bash:`-fte`
    TERMINATION_THRESHOLD_FACTOR       :bash:`--termination-threshold-factor`      :bash:`-ttf`
    TERMINATION_DIMINISHING_FACTOR     :bash:`--termination-diminishing-factor`    :bash:`-tdf`
    LR                                 :bash:`--lr`                                :bash:`-l`
    BATCH_SIZE                         :bash:`--batch-size`                        :bash:`-bs`
    OPTIMIZER                          :bash:`--optimizer`                         :bash:`-o`
    AFFINE_TRANSFORMATIONS_ENABLED     :bash:`--affine-transformations-enabled`    :bash:`-ate`
    CUTOUT_PROB                        :bash:`--cutout-prob`                       :bash:`-cp`
    SATURATION_AUG_PROB                :bash:`--saturation-aug-prob`               :bash:`-sap`
    CONTRAST_AUG_PROB                  :bash:`--contrast-aug-prob`                 :bash:`-cap`
  ==================================  ==========================================  ===============