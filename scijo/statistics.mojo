from tensor import Tensor, TensorShape
import math

from algorithm import vectorize, parallelize
from builtin.dtype import DType

# struct pInf[T:DType]():
#     var value:Scalar[T] 

#     fn __init__(inout self):
#         self.value = 0.0

#     fn __lt__(self, other:Scalar[T])->Bool:
#         return False
    
#     fn __le__(self, other:Scalar[T])->Bool:
#         return False
    
#     fn __gt__(self, other:Scalar[T])->Bool:
#         return True
    
#     fn __ge__(self, other:Scalar[T])->Bool:
#         return True

#     fn __eq__(self, other:Scalar[T])->Bool:
#         return self.value == other.value
    
#     fn __ne__(self, other:Scalar[T])->Bool:
#         return self.value != other.value

#     fn __str__(self)->String:
#         return "+Inf"
    
#     fn print(self)->None:
#         print("+Inf")

# struct mInf[T:DType]():
#     var value:Scalar[T] 

#     fn __init__(inout self):
#         self.value = 0.0

#     fn __lt__(self, other:mInf[T])->Bool:
#         return True
    
#     fn __le__(self, other:mInf[T])->Bool:
#         return True
    
#     fn __gt__(self, other:mInf[T])->Bool:
#         return False
    
#     fn __ge__(self, other:mInf[T])->Bool:
#         return False

#     fn __eq__(self, other:mInf[T])->Bool:
#         if other.value == math.limit.inf[T]:
#             return True
#         return False
    
#     fn __ne__(self, other:mInf[T])->Bool:
#         if other.value != math.limit.inf[T]:
#             return True
#         return False

#     fn __str__(self)->String:
#         return "-Inf"
    
#     fn print(self)->None:
#         print("-Inf")


# SORTING ALGORITHMS
fn binary_sort[T:DType](tensor:Tensor[T])->Tensor[T]:
    var result:Tensor[T] = tensor
    var n = tensor.num_elements()
    for end in range(n, 1, -1):
        for i in range(1, end):
            if result[i-1] > result[i]:
                var temp = result[i-1]
                result[i-1] = result[i]
                result[i] = temp
    return result
 
### NON MATH LIBRARY FUNCTIONS
# ! All these functions only calculate for 1D tensor, I need to implement them for ND using input axis parameter like numpy
fn sum[T:DType](tensor:Tensor[T])->Scalar[T]:
    var result = Scalar[T]()
    alias simd_width: Int = simdwidthof[T]()
    @parameter
    fn vectorize_sum[simd_width: Int](idx: Int) -> None:
        result += tensor.data().load[width = simd_width](idx).reduce_add()
    vectorize[vectorize_sum, simd_width](tensor.num_elements())
    return result

fn prod[T:DType](tensor:Tensor[T])->Scalar[T]:
    var result = Scalar[T]()
    alias simd_width: Int = simdwidthof[T]()
    @parameter
    fn vectorize_mul[simd_width: Int](idx: Int) -> None:
        result *= tensor.data().load[width = simd_width](idx).reduce_mul()
    vectorize[vectorize_mul, simd_width](tensor.num_elements())
    return result

fn mean[T:DType](tensor:Tensor[T])->Scalar[T]:
    return sum[T](tensor) / tensor.num_elements()

fn mode[T:DType](tensor:Tensor[T])->Scalar[T]:
    var sorted_tensor = binary_sort[T](tensor)
    var max_count = 0
    var mode_value = sorted_tensor[0]
    var current_count = 1

    for i in range(1, tensor.num_elements()):
        if sorted_tensor[i] == sorted_tensor[i - 1]:
            current_count += 1
        else:
            if current_count > max_count:
                max_count = current_count
                mode_value = sorted_tensor[i - 1]
            current_count = 1

    if current_count > max_count:
        mode_value = sorted_tensor[tensor.num_elements() - 1]

    return mode_value

# * IMPLEMENT median high and low
fn median[T:DType](tensor:Tensor[T])->Scalar[T]:
    var sorted_tensor = binary_sort[T](tensor)
    var n = tensor.num_elements()
    if n % 2 == 1:
        return sorted_tensor[n // 2]
    else:
        return (sorted_tensor[n // 2 - 1] + sorted_tensor[n // 2]) / 2



fn max_vec[T:DType](tensor:Tensor[T])->Scalar[T]:
    # var max_value = Tensor[T](TensorShape(2), List[Scalar[T]](tensor.__getitem__(0),tensor.__getitem__(1)))
    # var max_value = Tensor[T](TensorShape(5), List[Scalar[T]](tensor[0], tensor[0], tensor[0], tensor[0], tensor[0]))
    alias nelts = simdwidthof[T]()
    var max_value = Tensor[T](TensorShape(nelts))
    for i in range(nelts):
        max_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        max_value.data().store[width=simd_width](0, math.max(max_value.data().load[width=simd_width](0), tensor.data().load[width=simd_width](idx)))
    vectorize[vectorized, nelts](tensor.num_elements())
    return max_value[0]



fn min_vec[T:DType](tensor:Tensor[T])->Scalar[T]:
    # var max_value = Tensor[T](TensorShape(2), List[Scalar[T]](tensor.__getitem__(0),tensor.__getitem__(1)))
    # var max_value = Tensor[T](TensorShape(5), List[Scalar[T]](tensor[0], tensor[0], tensor[0], tensor[0], tensor[0]))
    alias nelts = simdwidthof[T]()
    var max_value = Tensor[T](TensorShape(nelts))
    for i in range(nelts):
        max_value[i] = tensor[0]

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        max_value.data().store[width=simd_width](0, math.min(max_value.data().load[width=simd_width](0), tensor.data().load[width=simd_width](idx)))
    vectorize[vectorized, nelts](tensor.num_elements())
    return max_value[0]

fn pvariance[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    var mean_value:Scalar[T]

    if mu == Scalar[T]():
        mean_value = mean[T](tensor)
    else:
        mean_value = mu

    var sum = Scalar[T]()
    for i in range(tensor.num_elements()):
        sum += (tensor[i] - mean_value) ** 2
    return sum / tensor.num_elements()

fn variance[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    var mean_value:Scalar[T]

    if mu == Scalar[T]():
        mean_value = mean[T](tensor)
    else:
        mean_value = mu

    var sum = Scalar[T]()
    for i in range(tensor.num_elements()):
        sum += (tensor[i] - mean_value) ** 2
    return sum / (tensor.num_elements() -1)

fn pstdev[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    return math.sqrt(pvariance(tensor, mu))

fn stdev[T:DType](tensor:Tensor[T], mu:Scalar[T]=Scalar[T]())->Scalar[T]:
    return math.sqrt(variance(tensor, mu))

