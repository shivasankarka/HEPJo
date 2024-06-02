from tensor import Tensor
import math

# TODO: Vectorize this function
fn trapz[T: DType](inout x: Tensor[T], inout y: Tensor[T]) -> Float64:
    var integral: Float64 = 0.0
    for i in range(x.__len__() - 1):
        integral += (x[i] + x[i + 1]) * (y[i] + y[i + 1]) / 2.0
    return integral

