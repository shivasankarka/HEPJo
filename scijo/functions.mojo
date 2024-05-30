from scijo.vector import Vector3D
from tensor import Tensor
from builtin.dtype import DType

from algorithm import vectorize
from builtin.dtype import DType
import math

# TODO: Vectorize this function
fn trapz(inout x: Vector3D[DType.float64], inout y: Vector3D[DType.float64]) -> Float64:
    var integral: Float64 = 0.0
    for i in range(x.__len__() - 1):
        integral += (x[i] + x[i + 1]) * (y[i] + y[i + 1]) / 2.0
    return integral

# fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self: Tensor[DType.float64]) -> Tensor[DType.float64]:
#     alias simd_width: Int = simdwidthof[Float64]()
#     var new_array = Tensor[DType.float64](self.num_elements())
#     @parameter
#     fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
#         new_array._ptr.store[width=simd_width](idx, func[Float64, simd_width](self._ptr.load[width=simd_width](idx)))
#     vectorize[elemwise_vectorize, simd_width](self.num_elements())
#     return new_array

# fn sin(tens: Tensor[DType.float64]) -> Tensor[DType.float64]:
#     return _elementwise_scalar_arithmetic[sin](tens)

fn _math_func[dtype:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor: Vector3D[dtype])->Vector3D[dtype]:
    var result_tensor: Vector3D[dtype] = Vector3D[dtype]()
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        result_tensor._ptr.store[width=simd_width](idx, func[dtype, simd_width](tensor._ptr.load[width=simd_width](idx)))
    
    vectorize[vectorized, opt_nelts](tensor.__len__())
    return result_tensor

fn sin[dtype:DType](tensor:Vector3D[dtype])->Vector3D[dtype]:
    return _math_func[dtype,math.sin](tensor)

fn cos[dtype:DType](tensor:Vector3D[dtype])->Vector3D[dtype]:
    return _math_func[dtype,math.cos](tensor)

fn tan[dtype:DType](tensor:Vector3D[dtype])->Vector3D[dtype]:
    return _math_func[dtype,math.tan](tensor)

fn asin[dtype:DType](tensor:Vector3D[dtype])->Vector3D[dtype]:
    return _math_func[dtype,math.asin](tensor)

fn erf[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.erf](tensor)

fn identity[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.identity](tensor)

fn erfc[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.erfc](tensor)

fn lgamma[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.lgamma](tensor)

fn tgamma[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.tgamma](tensor)

fn nearbyint[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.nearbyint](tensor)

fn rint[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.rint](tensor)

fn j0[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.j0](tensor)

fn j1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.j1](tensor)

fn y0[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.y0](tensor)

fn y1[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.y1](tensor)

fn ulp[dtype:DType](tensor:Tensor[dtype])->Tensor[dtype]:
    return _math_func[dtype,math.ulp](tensor)
