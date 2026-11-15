"""Classification of glucocorticoid-responsive enhancers in A549 cells.

Core modules of the modelling pipeline:

``config``                  all settings: label thresholds, cross-validation, model parameters
``paths``                   local data and result folders, read from the git-ignored ``paths.toml``
``data_loading``            input tables -> raw feature matrix
``labels``                  UP / DN / NR classes and the three pairwise tasks
``feature_engineering``     motif indicators, optional dex/control contrasts
``preprocessing``           fold-fitted variance filter and categorical handling
``models``                  CatBoost and random-forest fits
``thresholding``            decision threshold from inner cross-validation
``evaluation``              held-out metrics
``permutation_importance``  held-out permutation importance
``stability``               features that are important in most folds
``final_model``             models refitted on all enhancers for interpretation
``intervals``               overlap of enhancers with peak files
``compat``                  loading models saved by the pre-release layout of this code
"""

__version__ = "1.0.0"
