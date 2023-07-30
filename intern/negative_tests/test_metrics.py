import metrics


def test_non_int_clicks():
    """Test for non int clicks"""
    try:
        metrics.ctr(1.5, 2)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    """Test for non int views."""
    try:
        metrics.ctr(3, 1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("Non int views not handled")


def test_non_positive_clicks():
    """Test for positive clicks."""
    try:
        metrics.ctr(-4, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Negatived clicks not handled")


def test_non_positive_views():
    """Test for positive views."""
    try:
        metrics.ctr(4, -2)
    except ValueError:
        pass
    else:
        raise AssertionError("Negatived views not handled")


def test_clicks_greater_than_views():
    """Test for clicks greater that views"""
    try:
        metrics.ctr(3, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("""Clicks greater than views not handled""")



def test_zero_views():
    """Test for zero division"""
    try:
        metrics.ctr(3, 0)
    except ZeroDivisionError:
        pass
    except ValueError:
        pass
    else:
        raise AssertionError("""Zero views not handeled or
                             views lesser than clicks""")
