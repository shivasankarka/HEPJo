from testing.testing import assert_true, assert_false
from hepjo import * 

def test_constructor1():
    var lv = LorentzVector[i32]()
    assert_true(lv[0] == 0, "Default constructor failed")
    assert_true(lv[1] == 0, "Default constructor failed")
    assert_true(lv[2] == 0, "Default constructor failed")
    assert_true(lv[3] == 0, "Default constructor failed")

def test_constructor_with_data():
    var lv = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    assert_true(lv[0] == 1.0, "Constructor with data failed")
    assert_true(lv[1] == 2.0, "Constructor with data failed")
    assert_true(lv[2] == 3.0, "Constructor with data failed")
    assert_true(lv[3] == 4.0, "Constructor with data failed")

def test_equality():
    var lv1 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var lv2 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var lv3 = LorentzVector[f64](5.0, 6.0, 7.0, 8.0)
    for i in range(4):
        assert_true(lv1[i] == lv2[i], "Equality test failed")
        assert_false(lv1[i] == lv3[i], "Equality test failed")

def test_inequality():
    var lv1 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var lv2 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var lv3 = LorentzVector[f64](5.0, 6.0, 7.0, 8.0)
    for i in range(4):
        assert_false(lv1[i] != lv2[i], "Inequality test failed")
        assert_true(lv1[i] != lv3[i], "Inequality test failed")

def test_length():
    var lv = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    assert_true(len(lv) == 4, "Length test failed")

# def test_iterator():
#     var lv = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
#     var sum = 0.0
#     for val in lv:
#         sum += val
#     assert_true(sum == 10.0, "Iterator test failed")

def test_dot_product():
    var vec1 = Vector3D[f64](1.0, 2.0, 3.0)
    var vec2 = Vector3D[f64](4.0, 5.0, 6.0)
    var dot_product = vec1.dot(vec2)
    assert_true(dot_product == 32.0, "Dot product test failed")

def test_addition():
    var lv1 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var lv2 = LorentzVector[f64](5.0, 6.0, 7.0, 8.0)
    var result = lv1 + lv2
    assert_true(result[0] == 6.0, "Addition test failed")
    assert_true(result[1] == 8.0, "Addition test failed")
    assert_true(result[2] == 10.0, "Addition test failed")
    assert_true(result[3] == 12.0, "Addition test failed")

def test_subtraction():
    var lv1 = LorentzVector[f64](5.0, 6.0, 7.0, 8.0)
    var lv2 = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var result = lv1 - lv2
    assert_true(result[0] == 4.0, "Subtraction test failed")
    assert_true(result[1] == 4.0, "Subtraction test failed")
    assert_true(result[2] == 4.0, "Subtraction test failed")
    assert_true(result[3] == 4.0, "Subtraction test failed")

def test_multiplication():
    var lv = LorentzVector[f64](1.0, 2.0, 3.0, 4.0)
    var scalar = 2.0
    var result = lv * scalar
    assert_true(result[0] == 2.0, "Multiplication test failed")
    assert_true(result[1] == 4.0, "Multiplication test failed")
    assert_true(result[2] == 6.0, "Multiplication test failed")
    assert_true(result[3] == 8.0, "Multiplication test failed")

def test_division():
    var lv = LorentzVector[f64](4.0, 6.0, 8.0, 10.0)
    var scalar = 2.0
    var result = lv / scalar
    assert_true(result[0] == 2.0, "Division test failed")
    assert_true(result[1] == 3.0, "Division test failed")
    assert_true(result[2] == 4.0, "Division test failed")
    assert_true(result[3] == 5.0, "Division test failed")

def test_magnitude():
    var lv = LorentzVector[f64](3.0, 4.0, 0.0, 0.0)
    var magnitude = lv.mag()
    assert_true(magnitude == 5.0, "Magnitude test failed")

def test_dot1():
    var lv1 = LorentzVector[f64, -1](1.0, 2.0, 3.0, 4.0)
    var lv2 = LorentzVector[f64, -1](5.0, 6.0, 7.0, 8.0)
    var dot_product = lv1.dot(lv2)
    assert_true(dot_product == 60.0, "Dot product test failed")

def test_dot2():
    var lv1 = LorentzVector[f64, 1](1.0, 2.0, 3.0, 4.0)
    var lv2 = LorentzVector[f64, 1](5.0, 6.0, 7.0, 8.0)
    var dot_product = lv1.dot(lv2)
    assert_true(dot_product == -60.0, "Dot product test failed")