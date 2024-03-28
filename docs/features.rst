.. _features:

Overview of available features
==============================

.. div:: dropdown-group

    .. dropdown:: Data Sources

        :class:`~qsprpred.data.sources.data_source.DataSource`: Base class for data sources.

        Data sources are used to load data from a source programmatically.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.sources.papyrus.papyrus_class.Papyrus`: Papyrus (See `data collection with Papyrus tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/data/data_collection_with_papyrus.ipynb>`_.)

    .. dropdown:: Data Filters

        :class:`~qsprpred.data.processing.data_filters.DataFilter`: Base class for data filters.

        Data filters are used to filter data based on some criteria.
        Examples can be found in the `data preparation tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/data/data_preparation.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core
                
                * :class:`~qsprpred.data.processing.data_filters.CategoryFilter`: CategoryFilter
                * :class:`~qsprpred.data.processing.data_filters.RepeatsFilter`: RepeatsFilter

    .. dropdown:: Descriptor Sets

        :class:`~qsprpred.data.descriptors.sets.DescriptorSet`: Base class for descriptor sets.

        Descriptor sets are used to calculate molecular descriptors for a set of molecules.
        Examples can be found in the `descriptor calculation tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/data/descriptors.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.descriptors.sets.DrugExPhyschem`: DrugExPhyschem 
                * :class:`~qsprpred.data.descriptors.sets.PredictorDesc`:PredictorDesc 
                * :class:`~qsprpred.data.descriptors.sets.RDKitDescs`: RDKitDescs
                * :class:`~qsprpred.data.descriptors.sets.SmilesDescs`: SmilesDescs
                * :class:`~qsprpred.data.descriptors.sets.TanimotoDistances`: TanimotoDistances
                * :class:`~qsprpred.data.descriptors.sets.DataFrameDescriptorSet`: DataFrameDescriptorSet
                * :class:`~qsprpred.data.descriptors.fingerprints.Fingerprint`: Fingerprint
                    * :class:`~qsprpred.data.descriptors.fingerprints.AtomPairFP`: AtomPairFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.AvalonFP`: AvalonFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.LayeredFP`: LayeredFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.MaccsFP`: MaccsFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.MorganFP`: MorganFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.PatternFP`: PatternFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.RDKitFP`: RDKitFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.RDKitMACCSFP`: RDKitMACCSFP
                    * :class:`~qsprpred.data.descriptors.fingerprints.TopologicalFP`: TopologicalFP

            .. tab-item:: Extra

                * :class:`~qsprpred.extra.data.descriptors.sets.ExtendedValenceSignature`: ExtendedValenceSignature
                * :class:`~qsprpred.extra.data.descriptors.sets.Mold2`: Mold2
                * :class:`~qsprpred.extra.data.descriptors.sets.Mordred`: Mordred
                * :class:`~qsprpred.extra.data.descriptors.sets.PaDEL`: PaDEL
                * :class:`~qsprpred.extra.data.descriptors.sets.ProteinDescriptorSet`: ProteinDescriptorSet
                    * :class:`~qsprpred.extra.data.descriptors.sets.ProDec`: ProDec
                * :class:`~qsprpred.data.descriptors.fingerprints.Fingerprint`: Fingerprint
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKAtomPairs2DFP`: CDKAtomPairs2DFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKEStateFP`: CDKEStateFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKExtendedFP`: CDKExtendedFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKFP`: CDKFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKGraphOnlyFP`: CDKGraphOnlyFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKKlekotaRothFP`: CDKKlekotaRothFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKMACCSFP`: CDKMACCSFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKPubchemFP`: CDKPubchemFP
                    * :class:`~qsprpred.extra.data.descriptors.fingerprints.CDKSubstructureFP`: CDKSubstructureFP

    .. dropdown:: Data Splitters

        :class:`~qsprpred.data.sampling.splits.DataSplit`: Base class for data splitters.

        Data splitters are used to split data into training and test sets.
        Examples can be found in the `data splitting tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/data/data_splitting.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.sampling.splits.RandomSplit`: RandomSplit
                * :class:`~qsprpred.data.sampling.splits.ScaffoldSplit`: ScaffoldSplitter
                * :class:`~qsprpred.data.sampling.splits.TemporalSplit`: StratifiedSplitter
                * :class:`~qsprpred.data.sampling.splits.ManualSplit`: ManualSplit
                * :class:`~qsprpred.data.sampling.splits.BootstrapSplit`: BootstrapSplit
                * :class:`~qsprpred.data.sampling.splits.GBMTDataSplit`: GBMTDataSplit
                    * :class:`~qsprpred.data.sampling.splits.GBMTRandomSplit`: GBMTRandomSplit
                    * :class:`~qsprpred.data.sampling.splits.ClusterSplit`: ClusterSplit

            .. tab-item:: Extra

                * :class:`~qsprpred.extra.data.sampling.splits.LeaveTargetsOut`: LeaveTargetsOut
                * :class:`~qsprpred.extra.data.sampling.splits.PCMSplit`: PCMSplit
                    * :class:`~qsprpred.extra.data.sampling.splits.TemporalPerTarget`: TemporalPerTarget


    .. dropdown:: Feature Filters

        :class:`~qsprpred.data.processing.feature_filters.FeatureFilter`: Base class for feature filters.

        Feature filters are used to filter features based on some criteria.
        Examples can be found in the `data preparation tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/data/data_preparation.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.processing.feature_filters.HighCorrelationFilter`: HighCorrelationFilter
                * :class:`~qsprpred.data.processing.feature_filters.LowVarianceFilter`: LowVarianceFilter
                * :class:`~qsprpred.data.processing.feature_filters.BorutaFilter`: BorutaFilter (:code:`numpy` version restricted to :code:`numpy<1.24.0`)

    .. dropdown:: Models

        :class:`~qsprpred.models.model.QSPRModel`: Base class for models.

        Models are used to predict properties of molecules.
        A general example can be found in the `quick start tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/quick_start.ipynb>`_.
        More detailed information can be found throughout the basic and advanced modelling tutorials.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.scikit_learn.SklearnModel`: SklearnModel

            .. tab-item:: Extra

                * :class:`~qsprpred.extra.models.pcm.PCMModel`: PCMModel (See `PCM tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/advanced/modelling/PCM_modelling.ipynb>`_.)

            .. tab-item:: GPU
                
                More information can be found in the `deep learning tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/advanced/modelling/deep_learning_models.ipynb>`_.
                
                * :class:`~qsprpred.extra.gpu.models.dnn.DNNModel`: DNNModel
                * :class:`~qsprpred.extra.gpu.models.chemprop.ChempropModel`: ChempropModel (See `Chemprop tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/advanced/modelling/chemprop_models.ipynb>`_.)
                * :class:`~qsprpred.extra.gpu.models.pyboost.PyBoostModel`: PyBoostModel

    .. dropdown:: Metrics

        :class:`~qsprpred.models.metrics.Metric`: Base class for metrics

        Metrics are used to evaluate the performance of models.
        More information can be found in the `model assessment tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/modelling/model_assessment.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.metrics.SklearnMetrics`: SklearnMetrics

    .. dropdown:: Model Assessors

        :class:`~qsprpred.models.assessment.methods.ModelAssessor`: Base class for model assessors.

        Model assessors are used to assess the performance of models.
        More information be found in the `model assessment tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/basics/modelling/model_assessment.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.assessment.methods.CrossValAssessor`: CrossValAssessor
                * :class:`~qsprpred.models.assessment.methods.TestSetAssessor`: TestSetAssessor

    .. dropdown:: Hyperparameter Optimizers

        :class:`~qsprpred.models.hyperparam_optimization.HyperparameterOptimization`: Base class for hyperparameter optimizers.

        Hyperparameter optimizers are used to optimize the hyperparameters of models.
        More information can be found in the `hyperparameter optimization tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/advanced/modelling/hyperparameter_optimization.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.hyperparam_optimization.GridSearchOptimization`: GridSearchOptimization
                * :class:`~qsprpred.models.hyperparam_optimization.OptunaOptimization`: OptunaOptimization


    .. dropdown:: Model Plots

        :class:`~qsprpred.plotting.base_plot.ModelPlot`: Base class for model plots.

        Model plots are used to visualize the performance of models.
        Examples can be found throughout the basic and advanced modelling tutorials.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.plotting.regression.RegressionPlot`: RegressionPlot
                    * :class:`~qsprpred.plotting.regression.CorrelationPlot`: CorrelationPlot
                    * :class:`~qsprpred.plotting.regression.WilliamsPlot`: WilliamsPlot
                * :class:`~qsprpred.plotting.classification.ClassifierPlot`: ClassifierPlot
                    * :class:`~qsprpred.plotting.classification.ROCPlot`: ROCPlot
                    * :class:`~qsprpred.plotting.classification.PRCPlot`: PRCPlot
                    * :class:`~qsprpred.plotting.classification.CalibrationPlot`: CalibrationPlot
                    * :class:`~qsprpred.plotting.classification.MetricsPlot`: MetricsPlot
                    * :class:`~qsprpred.plotting.classification.ConfusionMatrixPlot`: ConfusionMatrixPlot

    .. dropdown:: Monitors

        * :class:`~qsprpred.models.monitors.FitMonitor`: Base class for monitoring model fitting
        * :class:`~qsprpred.models.monitors.AssessorMonitor`: Base class for monitoring model assessment (subclass of :class:`~qsprpred.models.monitors.FitMonitor`)
        * :class:`~qsprpred.models.monitors.HyperparameterOptimizationMonitor`: Base class for monitoring hyperparameter optimization (subclass of :class:`~qsprpred.models.monitors.AssessorMonitor`)

        Monitors are used to monitor the training of models.
        More information can be found in the `model monitoring tutorial <https://github.com/CDDLeiden/QSPRpred/blob/main/tutorials/advanced/modelling/monitoring.ipynb>`_.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.monitors.NullMonitor`: NullMonitor
                * :class:`~qsprpred.models.monitors.ListMonitor`: ListMonitor
                * :class:`~qsprpred.models.monitors.BaseMonitor`: BaseMonitor
                    * :class:`~qsprpred.models.monitors.FileMonitor`: FileMonitor
                    * :class:`~qsprpred.models.monitors.WandBMonitor`: WandBMonitor

    .. dropdown:: Scaffolds

        :class:`~qsprpred.data.chem.scaffolds.Scaffold`: Base class for scaffolds.

        Class for calculating molecular scaffolds of different kinds

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.chem.scaffolds.Murcko`: Murcko
                * :class:`~qsprpred.data.chem.scaffolds.BemisMurcko`: BemisMurcko

    .. dropdown:: Clustering

        :class:`~qsprpred.data.chem.clustering.MoleculeClusters`: Base class for clustering molecules.

        Classes for clustering molecules

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.chem.clustering.RandomClusters`: RandomClusters
                * :class:`~qsprpred.data.chem.clustering.ScaffoldClusters`: ScaffoldClusters
                * :class:`~qsprpred.data.chem.clustering.FPSimilarityClusters`: FPSimilarityClusters
                    * :class:`~qsprpred.data.chem.clustering.FPSimilarityMaxMinClusters`: FPSimilarityMaxMinClusters
                    * :class:`~qsprpred.data.chem.clustering.FPSimilarityLeaderPickerClusters`: FPSimilarityLeaderPickerClusters
