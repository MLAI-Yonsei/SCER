import numpy as np
from scer.utils import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    IMAGE_DATASETS = ["Waterbirds", "CelebA", "MetaShift", "ImagenetBG", "NICOpp",
                      "MIMICNoFinding", "CXRMultisite", "CheXpertNoFinding",
                      "Living17", "Entity13", "Entity30", "Nonliving26", "CMNIST"]
    TEXT_DATASETS = ["CivilCommentsFine", "MultiNLI", "CivilComments"]
    TABULAR_DATASET = ["MIMICNotes"]

    HALF_BS_ALGOS = []

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert name not in hparams
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions

    _hparam('resnet18', False, lambda r: False)
    # nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False, lambda r: bool(r.choice([False, False])))

    if algorithm in ['GroupDRO', 'SCER','ERM']:
        _hparam('group_balanced', False, lambda r: True)
    else:
        _hparam('group_balanced', True, lambda r: False)

    # Algorithm-specific hparam definitions
    # Each block of code below corresponds to one algorithm

    if "GroupDRO" in algorithm:
        _hparam('groupdro_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))


    elif algorithm == "SCER":
        _hparam('scer_eta', 1e-2, lambda r: 10**r.uniform(-3, -1))
        _hparam('lambda_spurious', 1.0, lambda r: r.uniform(0, 1))  # 추가된 부분
        _hparam('lambda_core', 1.0, lambda r: r.uniform(0, 1))

    # Dataset-and-algorithm-specific hparam definitions
    # Each block of code below corresponds to exactly one hparam. Avoid nested conditionals

    if dataset in {"Living17", "Entity13", "Entity30", "Nonliving26"}:
        _hparam('pretrained', False, lambda r: False)
    else:
        _hparam('pretrained', True, lambda r: True)

    if dataset in TABULAR_DATASET:
        _hparam('mlp_width', 256, lambda r: int(2 ** r.uniform(6, 10)))
        _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))
        _hparam('mlp_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    if dataset in IMAGE_DATASETS + TABULAR_DATASET:
        _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4, -2))
    else:
        _hparam('lr', 1e-5, lambda r: 10**r.uniform(-5.5, -4))

    _hparam('weight_decay', 1e-4, lambda r: 10**r.uniform(-6, -3))

    if dataset in TEXT_DATASETS:
        _hparam('optimizer', 'adamw', lambda r: 'adamw')
    else:
        _hparam('optimizer', 'sgd', lambda r: 'sgd')

    if dataset in TEXT_DATASETS:
        _hparam('last_layer_dropout', 0.5, lambda r: r.choice([0., 0.1, 0.5]))
    else:
        _hparam('last_layer_dropout', 0., lambda r: 0.)

    if algorithm in HALF_BS_ALGOS:
        if dataset in TEXT_DATASETS:
            _hparam('batch_size', 16, lambda r: int(2**r.uniform(3, 4)))
        elif dataset in TABULAR_DATASET:
            _hparam('batch_size', 128, lambda r: int(2 ** r.uniform(7, 9)))
        else:
            _hparam('batch_size', 54, lambda r: int(2**r.uniform(5, 5.75)))
    else:
        if dataset in TEXT_DATASETS:
            _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))
        elif dataset in TABULAR_DATASET:
            _hparam('batch_size', 256, lambda r: int(2 ** r.uniform(7, 10)))
        else:
            _hparam('batch_size', 108, lambda r: int(2**r.uniform(6, 6.75)))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
