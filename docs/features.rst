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

        Dropdown content

    .. dropdown:: Feature Filters

        Dropdown content

    .. dropdown:: Models

        Dropdown content

    .. dropdown:: Metrics

        Dropdown content

    .. dropdown:: Model Assessors

        Dropdown content 

    .. dropdown:: Hyperparameter Optimizers

        Dropdown content

    .. dropdown:: Model Plots

        Dropdown content