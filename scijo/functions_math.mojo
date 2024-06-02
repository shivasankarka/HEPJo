from tensor import Tensor
import math

from algorithm import vectorize, parallelize
from builtin.dtype import DType
from testing import assert_raises

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

#######################################################
##### VECTORIZED FUNCTIONS WITH TWO INPUT TENSORS ######
#######################################################
fn _math_2Tinput_1Toutput[T: DType, func: fn[T: DType, simd_wid: Int] (SIMD[T, simd_wid], SIMD[T, simd_wid]) -> SIMD[T, simd_wid]]
    (t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    """
    Takes two tensor input and returns a new tensor with the transformed values with the specified SIMD-compatible function.

    Parameters:
        func: SIMD compatible math function
        T: Dtype of the input and output tensor.
        t1: The first input tensor.
        t2: The second input tensor.
    
    Returns:
        A new tensor with each element transformed by the specified SIMD-compatible function.
    """
    if t1.shape() != t2.shape():
        with assert_raises():
            raise Error("Shapes don't match, cannot apply the given function")

    var result_tensor: Tensor[T] = Tensor[T](t1.shape())
    alias nelts = simdwidthof[T]()

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        result_tensor.data().store[width=simd_width](idx, func[T, simd_width](t1.data().load[width=simd_width](idx), t2.data().load[width=simd_width](idx)))
    
    vectorize[vectorized, nelts](t1.num_elements())
    return result_tensor

fn add[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.add](t1, t2)

fn atan2[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.atan2](t1, t2)

fn div[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.div](t1, t2)

fn hypot[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.hypot](t1, t2)

fn mul[T:DType](t1: Tensor[T], t2: Tensor[T]) raises ->Tensor[T]:
    return _math_2Tinput_1Toutput[T,math.mul](t1, t2)

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

#######################################################
##### VECTORIZED FUNCTIONS WITH ONE INPUT TENSOR ######
#######################################################
# Is it possible to parallelize this by splitting the tensor into some N parts?
fn _math_1Tinput_1Toutput[T:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor: Tensor[T])->Tensor[T]:
    """
    Applies a SIMD-compatible function element-wise to the input tensor and returns a new tensor with the transformed values.

    Parameters:
        tensor: The input tensor to be transformed.
    
    Returns:
        A new tensor with each element transformed by the specified SIMD-compatible function.
    """
    var result_tensor: Tensor[T] = Tensor[T]()
    alias nelts = simdwidthof[T]()

    @parameter
    fn vectorized[simd_width: Int](idx: Int) -> None:
        result_tensor.data().store[width=simd_width](idx, func[T, simd_width](tensor.data().load[width=simd_width](idx)))
    
    vectorize[vectorized, nelts](tensor.num_elements())
    return result_tensor

# fn _math_1Tinput_1Toutput[T:DType, func: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](tensor: Tensor[T])->Tensor[T]:
    # """
    # Applies a SIMD-compatible function element-wise to the input tensor and returns a new tensor with the transformed values.

    # Parameters:
    #     tensor: The input tensor to be transformed.
    
    # Returns:
    #     A new tensor with each element transformed by the specified SIMD-compatible function.
    # """
    # var result_tensor: Tensor[T] = Tensor[T]()
    # alias nelts = simdwidthof[T]()

    # @parameter
    # fn maybe(m:Int) -> None:
    #     @parameter
    #     fn vectorized[simd_width: Int](idx: Int) -> None:
    #         result_tensor.data().store[width=simd_width](idx, func[T, simd_width](tensor.data().load[width=simd_width](idx)))
        
    #     vectorize[vectorized, nelts](tensor.num_elements())
    # parallelize[maybe]()
    # return result_tensor

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

