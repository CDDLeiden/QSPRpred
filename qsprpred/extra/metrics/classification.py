"""  Extra classification metrics from the RDKit.  """

import sklearn.metrics
from rdkit.ML.Scoring.Scoring import CalcEnrichment, CalcRIE, CalcBEDROC


def _bedroc_score(y_true, y_prob, alpha=20):
    """Calculate the Boltzmann-enhanced discrimination of ROC (BEDROC).

    Reference: Truchon and Bayly, J. Chem. Inf. Model. 2007 47 (2), 488-508. DOI: 10.1021/ci600426e

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_prob (np.array): Target probability scores. 1d array.
        alpha (int): weighting parameter (default: 20)

    Returns:
        float: The BEDROC.

    """
    return CalcBEDROC([[y] for _, y in sorted(zip(y_prob, y_true), reverse=True)],
                      col=0,
                      alpha=alpha)


def _enrichment_factor_score(y_true, y_prob, chi=0.05):
    """Calculate the enrichment factor.

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_prob (np.array): Target probability scores. 1d array.
        chi (int): weighting parameter (default: 5%)

    Returns:
        float: The enrichment factor.

    """
    return CalcEnrichment([[y] for _, y in sorted(zip(y_prob, y_true), reverse=True)],
                          col=0,
                          fractions=[chi])[0]

def _robust_initial_enhancement_score(y_true, y_prob, alpha=300):
    """Calculate the enrichment factor.

    Reference: Sheridan et al., J. Chem. Inf. Model. 2001 41 (5), 1395-1406. DOI: 10.1021/ci0100144

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_prob (np.array): Target probability scores. 1d array.
        alpha (int): weighting parameter (default: 100)

    Returns:
        float: The enrichment factor.

    """
    return CalcRIE([[y] for _, y in sorted(zip(y_prob, y_true), reverse=True)],
                   col=0,
                   alpha=alpha)


bedroc20_score = sklearn.metrics.make_scorer(_bedroc_score)
enrichment_factor5_score = sklearn.metrics.make_scorer(_enrichment_factor_score)
robust_initial_enhancement300_score = sklearn.metrics.make_scorer(_robust_initial_enhancement_score)
