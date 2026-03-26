import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    classes = np.unique(np.concatenate([y_pred,y_true]))

    tp_count = 0
    fp_count = 0
    fn_count = 0

    for c in classes:
        tp_count += np.sum((y_pred==c)&(y_true==c))
        fp_count +=np.sum((y_true!=c) & (y_pred==c))
        fn_count +=np.sum((y_pred!=c)&(y_true==c))

    return 2*tp_count/(2*tp_count+fp_count+fn_count)