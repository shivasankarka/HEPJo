from hepmojo.vector import Vector3D
from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

fn main() raises:
    var a = Tensor[DType.float32](TensorShape(3), List[Float32](1.0,2.0,3.0))
    var v = Vector3D(a)
    var v1 = Vector3D.frompoint(1.0,3.0,4.0)
    var v2 = Vector3D.fromvector(v1)
    var ten = Tensor[DType.float32](TensorShape(3), List[Float32](100.0,2.0,3.0))
    print(ten)
