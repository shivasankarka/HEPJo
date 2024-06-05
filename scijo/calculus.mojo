from tensor import Tensor
import math

from algorithm import vectorize, parallelize

# TODO: Vectorize this function 
# @parameter # example usage of reduce
# fn _reduce_sub[dtype:DType, width:Int](x: SIMD[dtype,width], y: SIMD[dtype,width]) -> SIMD[dtype,width]:
#     return y - x

#     var diff = x.data().load[width=2](idx).reduce[_reduce_sub, 1]()
#     var sum_val = (y.data().load[width=2](idx)).reduce_add()
#     print(diff, sum_val)

# I wonder if this can be parallelized
fn trapz[T: DType](y: Tensor[T], x: Tensor[T]) -> Scalar[T]:
    """
    Compute the integral of y over x using the trapezoidal rule.
    """
    var integral: Scalar[T] = 0.0
    alias nelts = simdwidthof[T]()

    for i in range(x.num_elements() - 1):
        integral += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0
    return integral


