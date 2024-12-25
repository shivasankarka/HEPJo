from testing.testing import assert_true, assert_equal
from hepjo import *


def test_constructor_1():
    var test1 = Vector2D()
    assert_true(test1[0] == 0, "default constructor failed")
    assert_true(test1[1] == 0, "default constructor failed")


def test_constructor_2():
    var test1 = Vector2D[i32](x=0, y=1)
    assert_true(test1[0] == 0, "default constructor failed")
    assert_true(test1[1] == 1, "default constructor failed")


def test_constructor_3():
    var test1 = Vector2D[i32](0, 1)
    assert_true(test1[0] == 0, "default constructor failed")
    assert_true(test1[1] == 1, "default constructor failed")


def test_constructor_4():
    var test1 = Vector2D[i32](1, 2)
    assert_true(test1[0] == 1, "default constructor failed")
    assert_true(test1[1] == 2, "default constructor failed")


def test_constructor_5():
    var test1 = List[Scalar[i32]](1, 2)
    var test2 = Vector2D[i32](test1)
    assert_true(test2[0] == 1, "default constructor failed")
    assert_true(test2[1] == 2, "default constructor failed")


def test_len():
    var test1 = Vector2D(1, 2)
    assert_true(len(test1) == 2, "__len__ doesn't work")


def test_str():
    var test1 = Vector2D[i32](1, 2)
    assert_true(
        str(test1) == "Vector2D: [1 , 2]" + "\n" + "dtype=int32",
        "__str__ doesn't work",
    )


def test_repr():
    var test1 = Vector2D[i32](1, 2)
    assert_true(
        test1.__repr__() == "Vector2D[DType.int32](x=1, y=2)",
        "__repr__ doesn't work",
    )


def test_iter():
    var test1 = Vector2D[i32](1, 2)
    var counter = 1
    for i in test1:
        assert_equal(i, counter, "iter failed")
        counter += 1


def test_addition():
    var result = Vector2D[i32](1, 2) + Vector2D[i32](3, 4)
    assert_equal[i32](result[0], 4, "addition failed at index 0")
    assert_equal[i32](result[1], 6, "addition failed at index 1")


def test_subtraction():
    var vec1 = Vector2D[i32](3, 4)
    var vec2 = Vector2D[i32](1, 2)
    var result = vec1 - vec2
    assert_true(result[0] == 2, "subtraction failed at index 0")
    assert_true(result[1] == 2, "subtraction failed at index 1")


def test_multiplication():
    var vec = Vector2D[i32](1, 2)
    var scalar = 2
    var result = vec * scalar
    assert_true(result[0] == 2, "multiplication failed at index 0")
    assert_true(result[1] == 4, "multiplication failed at index 1")


def test_division():
    var vec = Vector2D[i32](2, 4)
    var scalar = 2
    var result = vec / scalar
    assert_true(result[0] == 1, "division failed at index 0")
    assert_true(result[1] == 2, "division failed at index 1")


def test_dotproduct():
    var vec1 = Vector2D[i32](1, 2)
    var vec2 = Vector2D[i32](3, 4)
    var result = vec1.dot(vec2)
    assert_true(result == 11, "dot product failed")


def test_cross_product():
    # Cross product is not defined for 2D vectors
    pass


def test_magnitude():
    var vec = Vector2D[i32](3, 4)
    var result = vec.mag()
    assert_true(result == 5, "magnitude calculation failed")


def test_getitem():
    var vec = Vector2D[i32](1, 2)
    assert_true(vec[0] == 1, "getitem failed at index 0")
    assert_true(vec[1] == 2, "getitem failed at index 1")


def test_setitem():
    var vec = Vector2D[i32](1, 2)
    vec[0] = 3
    vec[1] = 4
    assert_true(vec[0] == 3, "setitem failed at index 0")
    assert_true(vec[1] == 4, "setitem failed at index 1")


def test_positive():
    var vec = Vector2D[i32](1, 2)
    var result = +vec
    assert_true(result[0] == 1, "positive failed at index 0")
    assert_true(result[1] == 2, "positive failed at index 1")


def test_negative():
    var vec = Vector2D[i32](1, 2)
    var result = -vec
    assert_true(result[0] == -1, "negative failed at index 0")
    assert_true(result[1] == -2, "negative failed at index 1")


def test_typeof():
    var vec = Vector2D[i32](1, 2)
    var result = vec.typeof()
    assert_true(result == DType.int32, "typeof failed")


def test_typeof_str():
    var vec = Vector2D[i32](1, 2)
    var result = vec.typeof_str()
    assert_true(result == "int32", "typeof_str failed")


def test_eq():
    var vec1 = Vector2D[i32](1, 2)
    var vec2 = Vector2D[i32](1, 2)
    var result = vec1 == vec2
    assert_true(result[0] == True, "eq failed at index 0")
    assert_true(result[1] == True, "eq failed at index 1")


def test_ne():
    var vec1 = Vector2D[i32](1, 2)
    var vec2 = Vector2D[i32](3, 4)
    var result = vec1 != vec2
    assert_true(result[0] == True, "ne failed at index 0")
    assert_true(result[1] == True, "ne failed at index 1")


def test_lt():
    var vec1 = Vector2D[i32](1, 2)
    var vec2 = Vector2D[i32](3, 4)
    var result = vec1 < vec2
    assert_true(result[0] == True, "lt failed at index 0")
    assert_true(result[1] == True, "lt failed at index 1")


def test_gt():
    var vec1 = Vector2D[i32](3, 4)
    var vec2 = Vector2D[i32](1, 2)
    var result = vec1 > vec2
    assert_true(result[0] == True, "gt failed at index 0")
    assert_true(result[1] == True, "gt failed at index 1")


def test_le():
    var vec1 = Vector2D[i32](1, 2)
    var vec2 = Vector2D[i32](3, 4)
    var result = vec1 <= vec2
    assert_true(result[0] == True, "le failed at index 0")
    assert_true(result[1] == True, "le failed at index 1")


def test_ge():
    var vec1 = Vector2D[i32](3, 4)
    var vec2 = Vector2D[i32](1, 2)
    var result = vec1 >= vec2
    assert_true(result[0] == True, "ge failed at index 0")
    assert_true(result[1] == True, "ge failed at index 1")
