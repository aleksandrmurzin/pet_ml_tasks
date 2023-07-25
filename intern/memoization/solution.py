from typing import Callable


def memoize(func: Callable) -> Callable:
    """
    Memoize function

    Parameters
    ----------
    func : Callable


    Returns
    -------


    """
    hash_talbe = dict()
    def wrapped(*args, **kwargs):
        """"""
        nonlocal hash_talbe

        hashed_func = hash(f"{str(args)}{str(kwargs)}")
        if hashed_func in hash_talbe:
            return hash_talbe[hashed_func]
        hash_talbe[hashed_func] = func(*args, **kwargs)
        return hash_talbe[hashed_func]
    return wrapped
