from utils import extend_matches



# p = [(1, 2), (7, 2)]
p = [(5, 3, 4, 8), (1, 2), (7, 2)]
exp = [(1, 2, 7), (3, 4, 5, 8)]

def test_extend_matches():
    assert exp == extend_matches(p)