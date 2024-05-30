# from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

# import scijo as sj
from testing import assert_equal
from scijo import Vector3D, LorentzVector

fn main() raises:
    # assert_equal[SIMD[DType.float32]](1.0, 1.0, msg="true")
    # var a = Tensor[DType.float32](TensorShape(3), List[Float32](1.0,2.0,3.0))
    # var v = Vector3D(a)
    # var v1 = Vector3D.frompoint(1.0,3.0,4.0)
    # var v2 = Vector3D.fromvector(v1)
    # print(v2.rho())
    # var sm = v1 / v2

    var arr3 = LorentzVector(1.0, 2.0, 3.0, 10.0)
    var boostvec = arr3.boostvector()
    print(arr3.__repr__())
    print(arr3.minv())
    print(arr3.isspacelike())
    arr3 = arr3.torestframe()
    print(arr3.__repr__())

    # var mat = sj.Vector3D(3, 3, 3)
    # var mat1 = sj.Vector3D(1, 2, 3)

    # # print(mat.__str__())
    # var arr = sj.array(3, 3)
    # arr = arr / 2.0
    # # print(arr)

    # var a = sj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # var b = sj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # print(a.__str__())
    # print(b.__str__())
    # var c = a + b
    # print(c.__str__())
