import metrics


def test_profit() -> None:
    """
    Tests profit function

    Args:

    Returns:

    Raises:

    
    """
    assert metrics.profit([1., 2., 3.], [1., 1., 1.]) == 3


def test_margin() -> None:
    """
    Tests margin function

    Args:

    Returns:

    Raises:

    
    """
    assert metrics.margin([1., 2., 3.], [0., 1., 2.]) == 0.5


def test_markup() -> None:
    """
    Test markup function

    Args:

    Returns:

    Raises:

    
    """
    assert metrics.markup([1., 2., 3.], [0., 1., 2.]) == 1.0
