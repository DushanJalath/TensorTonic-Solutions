import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape or fpr.ndim != 1 or len(fpr) < 2:
        return None

    return float(np.trapezoid(tpr,fpr))