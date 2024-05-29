# from hepmojo.vector import Vector3D
from hepmojo.vector_pure import Vector3D
from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

fn main() raises:
    # var a = Tensor[DType.float32](TensorShape(3), List[Float32](1.0,2.0,3.0))
    # var v = Vector3D(a)
    # var v1 = Vector3D.frompoint(1.0,3.0,4.0)
    # var v2 = Vector3D.fromvector(v1)
    # print(v2.rho())
    # var sm = v1 / v2

    var arr1 = Vector3D(1.0, 2.0, 3.0)
    var arr2 = Vector3D(4.0, 5.0, 6.0)

    # Normal and in-place
    var temp = arr1@arr2
    print("temp: ", temp.__str__())