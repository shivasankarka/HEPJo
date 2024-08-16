from testing.testing import assert_true, assert_equal
from hepjo import *

def test_constructor_1():
    var test1 = Vector3D()
    for i in range(3):
        assert_true(test1[i]==0, "default constructor failed")

def test_constructor_2():
    var test1 = Vector3D[i32](x=0, y=1, z=2)
    for i in range(3):
        assert_true(test1[i]==i, "default constructor failed")

def test_constructor_3():
    var test1 = Vector3D[i32](0, 1, 2)
    for i in range(3):
        assert_true(test1[i]==i, "default constructor failed")

def test_constructor_4():
    var test1 = Vector3D[i32](1, 2, 3)
    var test2 = Vector3D[i32](test1.data)
    for i in range(3):
        assert_true(test2[i]==i+1, "default constructor failed")

def test_constructor_5():
    var test1 = List[Scalar[i32]](1, 2, 3)
    var test2 = Vector3D[i32](test1)
    for i in range(3):
        assert_true(test2[i]==i+1, "default constructor failed")

def test_len():
    var test1 = Vector3D()
    assert_true(len(test1)==3, "__len__ doesn't work")

def test_str():
    var test1 = Vector3D[i32](1, 2, 3)
    assert_true(str(test1)=="Vector3D: [1 , 2 , 3]; dtype=int32", "__str__ doesn't work")

def test_repr():
    var test1 = Vector3D[i32](1, 2, 3)
    assert_true(test1.__repr__()=="Vector3D[DType.int32](x=1, y=2, z=3)", "__repr__ doesn't work")

def test_iter():
    var test1 = Vector3D[i32](1, 2, 3)
    var counter = 1
    for i in test1:
        assert_true(i == counter, "iter failed")
        counter += 1

def test_addition():
    var result = Vector3D[i32](1, 2, 3) + Vector3D[i32](4, 5, 6)
    assert_equal[i32](result[0], 5, "addition failed at index 0")
    assert_equal[i32](result[1], 7, "addition failed at index 1")
    assert_equal[i32](result[2], 9, "addition failed at index 2")

def test_subtraction():
    var vec1 = Vector3D[i32](4, 5, 6)
    var vec2 = Vector3D[i32](1, 2, 3)
    var result = vec1 - vec2
    assert_true(result[0] == 3, "subtraction failed at index 0")
    assert_true(result[1] == 3, "subtraction failed at index 1")
    assert_true(result[2] == 3, "subtraction failed at index 2")

def test_multiplication():
    var vec = Vector3D[i32](1, 2, 3)
    var scalar = 2
    var result = vec * scalar
    assert_true(result[0] == 2, "multiplication failed at index 0")
    assert_true(result[1] == 4, "multiplication failed at index 1")
    assert_true(result[2] == 6, "multiplication failed at index 2")

def test_division():
    var vec = Vector3D[i32](2, 4, 6)
    var scalar = 2
    var result = vec / scalar
    assert_true(result[0] == 1, "division failed at index 0")
    assert_true(result[1] == 2, "division failed at index 1")
    assert_true(result[2] == 3, "division failed at index 2")

def test_dotproduct():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](4, 5, 6)
    var result = vec1.dot(vec2)
    assert_true(result == 32, "dot product failed")

def test_cross_product():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](4, 5, 6)
    var result = vec1.cross(vec2)
    assert_true(result[0] == -3, "cross product failed at index 0")
    assert_true(result[1] == 6, "cross product failed at index 1")
    assert_true(result[2] == -3, "cross product failed at index 2")

def test_magnitude():
    var vec = Vector3D[i32](1, 2, 2)
    var result = vec.mag()
    assert_true(result == 3, "magnitude calculation failed")

def test_getitem():
    var vec = Vector3D[i32](1, 2, 3)
    assert_true(vec[0] == 1, "getitem failed at index 0")
    assert_true(vec[1] == 2, "getitem failed at index 1")
    assert_true(vec[2] == 3, "getitem failed at index 2")

def test_setitem():
    var vec = Vector3D[i32](1, 2, 3)
    vec[0] = 4
    vec[1] = 5
    vec[2] = 6
    assert_true(vec[0] == 4, "setitem failed at index 0")
    assert_true(vec[1] == 5, "setitem failed at index 1")
    assert_true(vec[2] == 6, "setitem failed at index 2")

def test_positive():
    var vec = Vector3D[i32](1, 2, 3)
    var result = +vec
    assert_true(result[0] == 1, "positive failed at index 0")
    assert_true(result[1] == 2, "positive failed at index 1")
    assert_true(result[2] == 3, "positive failed at index 2")

def test_negative():
    var vec = Vector3D[i32](1, 2, 3)
    var result = -vec
    assert_true(result[0] == -1, "negative failed at index 0")
    assert_true(result[1] == -2, "negative failed at index 1")
    assert_true(result[2] == -3, "negative failed at index 2")

def test_load():
    var vec = Vector3D[i32](1, 2, 3)
    assert_true(vec.load[width=1](0) == 1, "load failed at index 0")
    assert_true(vec.load[width=1](1) == 2, "load failed at index 1")
    assert_true(vec.load[width=1](2) == 3, "load failed at index 2")

def test_store():
    var vec = Vector3D[i32](1, 2, 3)
    vec.store(0, SIMD[i32, 1](4))
    vec.store(1, SIMD[i32, 1](5))
    vec.store(2, SIMD[i32, 1](6))
    assert_true(vec[0] == 4, "store failed at index 0")
    assert_true(vec[1] == 5, "store failed at index 1")
    assert_true(vec[2] == 6, "store failed at index 2")

def test_unsafe_ptr():
    var vec = Vector3D[i32](1, 2, 3)
    var result = vec.unsafe_ptr()
    assert_true(result[0] == 1, "unsafe_ptr failed at index 0")
    assert_true(result[1] == 2, "unsafe_ptr failed at index 1")
    assert_true(result[2] == 3, "unsafe_ptr failed at index 2")

def test_typeof():
    var vec = Vector3D[i32](1, 2, 3)
    var result = vec.typeof()
    assert_true(result == DType.int32, "typeof failed")

def test_typeof_str():
    var vec = Vector3D[i32](1, 2, 3)
    var result = vec.typeof_str()
    assert_true(result == "int32", "typeof_str failed")

def test_eq():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](1, 2, 3)
    var result = vec1 == vec2
    assert_true(result[0] == True, "eq failed at index 0")
    assert_true(result[1] == True, "eq failed at index 1")
    assert_true(result[2] == True, "eq failed at index 2")

def test_ne():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](4, 5, 6)
    var result = vec1 != vec2
    assert_true(result[0] == True, "ne failed at index 0")
    assert_true(result[1] == True, "ne failed at index 1")
    assert_true(result[2] == True, "ne failed at index 2")

def test_lt():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](4, 5, 6)
    var result = vec1 < vec2
    assert_true(result[0] == True, "lt failed at index 0")
    assert_true(result[1] == True, "lt failed at index 1")
    assert_true(result[2] == True, "lt failed at index 2")

def test_gt():
    var vec1 = Vector3D[i32](4, 5, 6)
    var vec2 = Vector3D[i32](1, 2, 3)
    var result = vec1 > vec2
    assert_true(result[0] == True, "gt failed at index 0")
    assert_true(result[1] == True, "gt failed at index 1")
    assert_true(result[2] == True, "gt failed at index 2")

def test_le():
    var vec1 = Vector3D[i32](1, 2, 3)
    var vec2 = Vector3D[i32](4, 5, 6)
    var result = vec1 <= vec2
    assert_true(result[0] == True, "le failed at index 0")
    assert_true(result[1] == True, "le failed at index 1")
    assert_true(result[2] == True, "le failed at index 2")

def test_ge():
    var vec1 = Vector3D[i32](4, 5, 6)
    var vec2 = Vector3D[i32](1, 2, 3)
    var result = vec1 >= vec2
    assert_true(result[0] == True, "ge failed at index 0")
    assert_true(result[1] == True, "ge failed at index 1")
    assert_true(result[2] == True, "ge failed at index 2")
