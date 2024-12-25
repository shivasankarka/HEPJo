from testing.testing import assert_true, assert_equal
from hepjo import *

def test_constructor_1():
    var test1 = Vector3D()
    for i in range(3):
        assert_true(test1[i]==0, "default constructor failed")