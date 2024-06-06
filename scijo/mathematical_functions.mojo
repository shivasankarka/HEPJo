from tensor import Tensor
import math

from algorithm import vectorize, parallelize
from builtin.dtype import DType
from testing import assert_raises

from .arrays import array
from .utils import *

### MATH FUNCTIONS NOT IMPLEMENTED FOR 2D T input
# equal
# greater
# greater_equal
# isclose

### MATH FUNCTIONS NOT IMPLEMENTED
# align_down
# align_down_residual
# align_up
# align_up_residual
# all_true
# any_true
# ceildiv
# clamp
# copys
# divceil
# divmod
# factorial
# fma
# frexp
# gcd
# iota
# is_even
# is_odd
# is_power_of_2
# isfinite
# isinf
# isnan
# lcm
# ldexp - args
# less - bool
# less_equal - bool
# logical_not# logical_a - bool
# logical_or - bool
# logical_xor - bool
# max - implement yourself
# min - implement yourself
# mod - implement yourself
# nan
# nonetrue - bool
# not_equal -  bool
# reduce_bit_count
# rotate_bits_left
# rotate_bits_right
# rotate_left
# rotate_right
# select

################################################
################ MATH FUNCTIONS ################
################################################

fn add[T:DType, calcmethod:CalcMethods = VectorizedMethod](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    # return t1 + t2 # technically this is already vectorized If I am not wrong and therefore should be the same speed. 
    return VectorizedMethod()._math_2Tinput_1Toutput[T,math.add](t1, t2)

fn atan2[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.atan2](t1, t2)

fn div[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.div](t1, t2)

fn hypot[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.hypot](t1, t2)

fn mul[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.mul](t1, t2)
    # return t1*t2

fn nextafter[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.nextafter](t1, t2)

fn power[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    # * TAKE CARE OF NAMING CONVENTION
    # THERE ARE FEW OTHER IMPLEMENTATIONS OF POWER FUNCTION, CHECK THAT IN DOCS
    return _math_2Tinput_1Toutput[T,math.pow](t1, t2)

fn remainder[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.remainder](t1, t2)

fn scalb[T:DType](tensor:Tensor[T], tensor2:Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.scalb](tensor, tensor2)

fn sub[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.sub](t1, t2)
    # return t1 - t2

fn acos[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.acos](tensor)

fn abs[T:DType](tensor:Tensor[T])->Tensor[T]:
    # ! CHECK IF THE NAME abs DOESN'T CLASH AND CREATE PROBLEMS
    return _math_1Tinput_1Toutput[T,math.abs](tensor)

fn acosh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.acosh](tensor)

fn asin[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.asin](tensor)

fn asinh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.asinh](tensor)

fn atan[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.atan](tensor)

fn atanh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.atanh](tensor)

fn cbrt[T:DType](tensor:Tensor[T])->Tensor[T]:
    # ! IMPLEMENT THE CONSTRAINT THAT DTYPE SHOULD BE FLOATING POINT
    return _math_1Tinput_1Toutput[T,math.cbrt](tensor)

fn ceil[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.ceil](tensor)

fn cos[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.cos](tensor)

fn cosh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.cosh](tensor)

fn erf[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.erf](tensor)

fn erfc[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.erfc](tensor)

fn exp[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.exp](tensor)

fn exp2[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.exp2](tensor)

fn expm1[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.expm1](tensor)

fn floor[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.floor](tensor)

fn identity[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.identity](tensor)

fn j0[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.j0](tensor)

fn j1[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.j1](tensor)

fn lgamma[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.lgamma](tensor)

fn log[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.log](tensor)

fn log10[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.log10](tensor)
    
fn log1p[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.log1p](tensor)

fn log2[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.log2](tensor)

fn logb[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.logb](tensor)

fn nearbyint[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.nearbyint](tensor)

fn reciprocal[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.reciprocal](tensor)

fn rint[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.rint](tensor)

fn round[T:DType](tensor:Tensor[T])->Tensor[T]:
    # CHECK NAMING CONVENTION
    return _math_1Tinput_1Toutput[T,math.round](tensor)

fn round_half_up[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.round_half_up](tensor)

fn round_half_down[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.round_half_down](tensor)

fn roundeven[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.roundeven](tensor)

fn rsqrt[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.rsqrt](tensor)

fn sin[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.sin](tensor)

fn sinh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.sinh](tensor)

fn sqrt[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.sqrt](tensor)

fn tan[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.tan](tensor)

fn tanh[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.tanh](tensor)

fn tgamma[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.tgamma](tensor)

fn trunc[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.trunc](tensor)

fn ulp[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.ulp](tensor)

fn y0[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.y0](tensor)

fn y1[T:DType](tensor:Tensor[T])->Tensor[T]:
    return _math_1Tinput_1Toutput[T,math.y1](tensor)

fn power[T:DType](tensor:Tensor[T], s: Scalar[T])->Tensor[T]:
    # return tensor**scalar
    return _math_1T1Sinput_1Toutput[T, math.pow](tensor, s)


##############################################
############ MAX & MIN #######################
##############################################

# TODO : Speed this up and reduce nan value checks
# Could check whole array for nan values using vectorization / parallelization and remove the conditions inside for loop, not sure if that's faster
fn max[T:DType](arr:Tensor[T])->Scalar[T]:
    var result_max:Scalar[T] = Scalar[T](arr[0])
    for i in range(arr.num_elements()):
        if math.isnan(arr.data().load[width=1](i)):
            result_max = arr.data().load[width=1](i)
        else:
            result_max = math.max(result_max[0], arr.data().load[width=1](i))
    return result_max

# TODO : Speed this up and reduce nan value checks 
# Could check whole array for nan values using vectorization / parallelization and remove the conditions inside for loop, not sure if that's faster
fn min[T:DType](arr:Tensor[T])->Scalar[T]:
    var result_min:Scalar[T] = Scalar[T](arr[0])
    for i in range(arr.num_elements()):
        if math.isnan(arr.data().load[width=1](i)):
            result_min = arr.data().load[width=1](i)
        else:
            result_min = math.max(result_min[0], arr.data().load[width=1](i))
    return result_min

# this seems to be just an alias for min in numpy
fn amin[T:DType](arr:Tensor[T])->Scalar[T]:
    return min[T](arr)

# this seems to be just an alias for max in numpy
fn amax[T:DType](arr:Tensor[T])->Scalar[T]:
    return max[T](arr)

fn mimimum[T:DType](s1:Scalar[T], s2:Scalar[T]) -> Scalar[T]:
    return math.max(s1, s2)

fn maximum[T:DType](s1:Scalar[T], s2:Scalar[T]) -> Scalar[T]:
    return math.max(s1, s2)

fn minimum[T:DType](tensor1:Tensor[T], tensor2:Tensor[T]) raises -> Tensor[T]:
    var result:Tensor[T] = Tensor[T](tensor1.shape())
    alias nelts = simdwidthof[T]()
    if tensor1.shape() != tensor2.shape():
        raise Error("Tensor shapes are not the same")
    
    @parameter
    fn vectorized_min[simd_width: Int](idx: Int) -> None:
        result.data().store[width = simd_width](idx, math.min(tensor1.data().load[width = simd_width](idx), tensor2.data().load[width = simd_width](idx)))
    
    vectorize[vectorized_min, nelts](tensor1.num_elements())
    return result

fn maximum[T:DType](tensor1:Tensor[T], tensor2:Tensor[T]) raises -> Tensor[T]:
    var result:Tensor[T] = Tensor[T](tensor1.shape())
    alias nelts = simdwidthof[T]()
    if tensor1.shape() != tensor2.shape():
        raise Error("Tensor shapes are not the same")
    
    @parameter
    fn vectorized_max[simd_width: Int](idx: Int) -> None:
        result.data().store[width = simd_width](idx, math.max(tensor1.data().load[width = simd_width](idx), tensor2.data().load[width = simd_width](idx)))
    
    vectorize[vectorized_max, nelts](tensor1.num_elements())
    return result

# * for loop version works fine, vectorized doesn't
fn argmax[T:DType](tensor:Tensor[T]) raises -> Int:
    if tensor.num_elements() == 0:
        raise Error("Tensor is empty")
    
    var idx:Int = 0
    var max_val:Scalar[T] = tensor[0]
    for i in range(1, tensor.num_elements()):
        if tensor[i] > max_val:
            max_val = tensor[i]
            idx = i
    return idx
    # alias nelts = simdwidthof[T]()
    # var max_val = Tensor[T](TensorShape(2), List[Scalar[T]](tensor.__getitem__(0), tensor.__getitem__(1)))
    # @parameter
    # fn vectorized_argmax[simd_width: Int](idx: Int) -> None:
    #     max_val.data().store[width=simd_width](0, math.max(tensor.data().load[width = simd_width](idx), max_val.data().load[width=simd_width](0)))
    # vectorize[vectorized_argmax, nelts](tensor.num_elements())
    # return idx

# * for loop version works fine, vectorized doesn't
fn argmin[T:DType](tensor:Tensor[T]) raises -> Int:
    if tensor.num_elements() == 0:
        raise Error("Tensor is empty")
    
    var idx:Int = 0
    var min_val:Scalar[T] = tensor[0]

    for i in range(1, tensor.num_elements()):
        if tensor[i] < min_val:
            min_val = tensor[i]
            idx = i
    return idx

    # alias nelts = simdwidthof[T]()
    # var max_val = Tensor[T](TensorShape(2), List[Scalar[T]](tensor.__getitem__(0), tensor.__getitem__(1)))
    # @parameter
    # fn vectorized_argmax[simd_width: Int](idx: Int) -> None:
    #     max_val.data().store[width=simd_width](0, math.min(tensor.data().load[width = simd_width](idx), max_val.data().load[width=simd_width](0)))
    # vectorize[vectorized_argmax, nelts](tensor.num_elements())
    # return idx

