.. _features:

Overview of available features
==============================

.. div:: dropdown-group

    .. dropdown:: Data Sources

        :class:`~qsprpred.data.sources.data_source.DataSource`: Base class for data sources.

        Data sources are used to load data from a source programmatically.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.sources.papyrus.papyrus_class.Papyrus`: Papyrus

    .. dropdown:: Data Filters

        :class:`~qsprpred.data.processing.data_filters.DataFilter`: Base class for data filters.

        Data filters are used to filter data based on some criteria.

        .. tab-set::

            .. tab-item:: Core
                
                * :class:`~qsprpred.data.processing.data_filters.CategoryFilter`: CategoryFilter
                * :class:`~qsprpred.data.processing.data_filters.RepeatsFilter`: RepeatsFilter

    .. dropdown:: Descriptor Sets

        :class:`~qsprpred.data.descriptors.sets.DescriptorSet`: Base class for descriptor sets.

        Descriptor sets are used to calculate molecular descriptors for a set of molecules.

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
                    * :class:`~qsprpred.data.descriptors.fingerprints.MACCsFP`: MACCsFP
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

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.data.processing.feature_filters.HighCorrelationFilter`: HighCorrelationFilter
                * :class:`~qsprpred.data.processing.feature_filters.LowVarianceFilter`: LowVarianceFilter
                * :class:`~qsprpred.data.processing.feature_filters.BorutaFilter`: BorutaFilter

    .. dropdown:: Models

        :class:`~qsprpred.models.models.QSPRModel`: Base class for models.

        Models are used to predict properties of molecules.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.scikit_learn.SklearnModel`: SklearnModel

            .. tab-item:: Extra

                * :class:`~qsprpred.extra.models.pcm.PCMModel`: PCMModel

            .. tab-item:: GPU
                    
                * :class:`~qsprpred.extra.gpu.models.chemprop.ChempropModel`: ChempropModel
                * :class:`~qsprpred.extra.gpu.models.pyboost.PyBoostModel`: PyBoostModel

    .. dropdown:: Metrics

        :class:`~qsprpred.models.metrics.Metric`: Base class for metrics

        Metrics are used to evaluate the performance of models.

        .. tab-set::

            .. tab-item:: Core

                * :class:`~qsprpred.models.metrics.SklearnMetrics`: SklearnMetrics

    .. dropdown:: Model Assessors

        Dropdown content 

    .. dropdown:: Hyperparameter Optimizers

        Dropdown content

    .. dropdown:: Model Plots

        Dropdown content