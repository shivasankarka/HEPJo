# from hepmojo.vector import Vector3D
from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

from hepmojo.vector_pure_traits import Vector3D, LorentzVector
from hepmojo.constants import h, c, pi

fn main() raises:
    # var a = Tensor[DType.float32](TensorShape(3), List[Float32](1.0,2.0,3.0))
    # var v = Vector3D(a)
    # var v1 = Vector3D.frompoint(1.0,3.0,4.0)
    # var v2 = Vector3D.fromvector(v1)
    # print(v2.rho())
    # var sm = v1 / v2

    var arr1 = Vector3D(1.0, 0.0, 0.0)
    var arr2 = Vector3D(0.0, 0.0, 1.0)
    var arr3 = LorentzVector(1.0, 2.0, 3.0, 4.0)
    var ang = arr1.cos_angle(arr2)
    print(arr1.__repr__())
    print(arr3.__repr__())
