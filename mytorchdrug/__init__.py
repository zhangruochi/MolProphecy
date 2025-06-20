_criterion_name = {
    "mse": "mean squared error",
    "mae": "mean absolute error",
    "bce": "binary cross entropy",
    "ce": "cross entropy",
    "focal": "focal loss",
}



def _get_criterion_name(criterion):
    if criterion in _criterion_name:
        return _criterion_name[criterion]
    return "%s loss" % criterion