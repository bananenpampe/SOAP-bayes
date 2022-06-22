import numpy as np
import sklearn.model_selection
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

def train_test_split(*arrays, **options):
    """This is an extended version of the sklearn train test split supporting
    overlapping train and test sets.
    See `sklearn.model_selection.train_test_split (external link)
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_ .
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        See
        `random state glossary from sklearn (external link) <https://scikit-learn.org/stable/glossary.html#term-random-state>`_
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    groups : array-like, default = None
        If not None, data is split keeping the groups together
    train_test_overlap : bool, default=False
        If True, and train and test set are both not None, the train and test
        set may overlap.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    """
    train_test_overlap = options.pop("train_test_overlap", False)
    test_size = options.get("test_size", None)
    train_size = options.get("train_size", None)


    if train_test_overlap and train_size is not None and test_size is not None:
        # checks from sklearn
        arrays = indexable(*arrays)
        n_samples = _num_samples(arrays[0])

        if test_size == 1.0 or test_size == n_samples:
            test_sets = arrays
        else:
            options["train_size"] = None
            test_sets = sklearn.model_selection.train_test_split(*arrays, **options)[
                1::2
            ]
            options["train_size"] = train_size

        if train_size == 1.0 or train_size == n_samples:
            train_sets = arrays
        else:
            options["test_size"] = None
            train_sets = sklearn.model_selection.train_test_split(*arrays, **options)[
                ::2
            ]
            options["test_size"] = test_size

        train_test_sets = []
        for i in range(len(train_sets)):
            train_test_sets += [train_sets[i], test_sets[i]]
        return train_test_sets
    else:
        return sklearn.model_selection.train_test_split(*arrays, **options)

from itertools import chain
from sklearn.utils import indexable, _safe_indexing

def atom_groups_by_frame(frames):
    """ Creates a list of group IDs associated with a series of ASE frames that
    point to the frame each atom center belongs to. If present, does the indexing
    referring only to the atoms selected by the `center_atoms_mask` array.
    Parameters
    ----------
    frames: list of ASE atom frames, used to infer the indexing of atoms to frames.
            all the atoms with `center_atom_mask[i] = True` are considered to be
            active - i.e. the targets will contain a list of all such atoms, in the
            same order as they appear in the frames.
    Returns
    -------
    groups: frame IDs for all the active atoms in the data set
    """

    groups = []
    for idx_f, f in enumerate(frames):
        if "center_atoms_mask" in f.arrays:
            n_atoms = np.count_nonzero(f.arrays["center_atoms_mask"])
        else:
            n_atoms = len(f.symbols)
        groups += [idx_f] * n_atoms

    return np.asarray(groups, dtype=int)

def train_test_split_by_frame(frames, *arrays, **options):
    """ Splits a dataset of atom-centered data into train and test sets along
    frame boundaries, using GroupShuffleSplit and atom_groups_by_frame to
    determine the list of frame IDs of each atomic center. Also returns a
    list of frame IDs for each of the train and test samples, that can be
    used to further keep the frames together in CV folding.
    Parameters
    ----------
    frames: list of ASE atom frames, used to infer frame indexing (see also
            `atom_groups_by_frame`
    *arrays, **options: arrays to perform the splitting, will be passed as
            arguments to GroupShuffleSplit
    Returns
    -------
    [train, test, ...], group_train, group_test :
            sequence of (train, test) splits of each array, plus the split of
            the frame IDs
    """

    groups = atom_groups_by_frame(frames)
    arrays += (groups,)
    arrays = indexable(*arrays)

    train_idx, test_idx = next(sklearn.model_selection.GroupShuffleSplit(n_splits=2, **options).split(arrays[0], arrays[1], groups))
    return list(
        chain.from_iterable(
            (_safe_indexing(a, train_idx), _safe_indexing(a, test_idx)) for a in arrays
        )
    )
