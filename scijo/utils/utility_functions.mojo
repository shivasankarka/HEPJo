# Defines the vectorized, parallelized weak internal use functions
from tensor import Tensor, TensorShape
from algorithm import vectorize, parallelize


trait CalcMethods():
    """
    Trait for all possible calculation methods to do mathematical computation in mojo
    for Tensors, Array (If defined in future)
    """

    fn __init__(inout self:Self):
        pass

    fn _mathfunc_2Tinput_1Toutput[T: DType, function: fn[T: DType, simd_wid: Int](SIMD[T, simd_wid], SIMD[T, simd_wid]) -> SIMD[T, simd_wid]](
        self, t1: Tensor[T], t2: Tensor[T]
    ) raises -> Tensor[T]:
    # Gotta create a own Tensor type instead and change this for better compatiblity.
    """
    Applies a SIMD-compatible function with two input tensors, t1 and t2, and returns a new tensor.

    Parameters:
        T: The data type of the input and output tensors.
        function: The SIMD-compatible function that operates on tensors.

    Args:
        t1: The first input tensor.
        t2: The second input tensor.

    Returns:
        A tensor resulting from the function applied to the input tensors as function(t1, t2).
    """
    ...

    fn _mathfunc_2Tinput_1Toutput[T: DType, function: fn[T: DType, simd_wid: Int] (SIMD[T, simd_wid], SIMD[T, simd_wid]) -> SIMD[T, simd_wid]](
        self, t1: Tensor[T], t2: Tensor[T]
    ) raises -> Tensor[T]:
    # Gotta create a own Tensor type instead and change this for better compatiblity.
    """
    Applies a SIMD-compatible function with two input tensors, t1 and t2, and returns a new tensor.

    Parameters:
        T: The data type of the input and output tensors.
        function: The SIMD-compatible function that operates on tensors.

    Args:
        t1: The first input tensor.
        t2: The second input tensor.

    Returns:
        A tensor resulting from the function applied to the input tensors as function(t1, t2).
    """
    ...

    fn _mathfunc_1Tinput_1Toutput[T:DType, function: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](
        self, t1: Tensor[T])->Tensor[T]:
    """
    Applies a SIMD-compatible function to the input tensor t1 and returns a new tensor.

    Parameters:
        T: The data type of the input and output tensors.
        function: The SIMD-compatible function that operates on tensors.

    Args:
        t1: The input tensor.

    Returns:
        A tensor resulting from the function applied to the input tensor as function(t1).
    """
    ...

    fn _mathfunc_1T1Sinput_1Toutput[T:DType, function: fn[T:DType, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[T, simd_w]](
        self, t1: Tensor[T], s: Int)->Tensor[T]:
    """
    Applies a SIMD-compatible function to the input tensor t1 and a scalar value s, returning a new tensor.

    Parameters:
        T: The data type of the input and output tensors.
        function: The SIMD-compatible function that operates on tensors.

    Args:
        t1: The input tensor of type T.
        s: An integer value of type Int.

    Returns:
        A tensor resulting from the function applied to the input tensor with the scalar value s, i.e., function(t1, s).
    """
    ...

    fn _mathfunc_2Tcompare[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t1: Tensor[T], t2: Tensor[T]) raises -> Tensor[DType.Bool]:
    """
    Applies a SIMD-compatible function to input tensor t1 and SIMD[T, 1], returning a tensor of type Bool.

    Parameters:
        T: Data type of the input and output tensors.
        simd_w: SIMD width of type T.
        function: SIMD-compatible function that operates on SIMD[T, simd_w] and outputs SIMD[DType.Bool, $1].

    Args:
        t1: Input tensor 1 of type T.
        t2: Input tensor 2 of type T.

    Returns:
        A tensor of type DType.Bool that results from applying the function to the input tensors as function(t1, t2).
    """
    ...

    fn _issomething[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t2: Tensor[T]) raises -> Tensor[DType.Bool]:
    """
    Determines if the input tensor t2 contains elements of a specific data type.

    Parameters:
        T: data type of the input tensor
        function: SIMD compatible function that acts on SIMD[T, simd_w] and returns SIMD[DType.Bool, simd_w]

    Args:
        t2: Input tensor of type T

    Returns:
        A tensor of type DType.Bool indicating whether the input tensor contains elements of the specified data type.
    """
    ...

struct VectorizedMethod[unroll_factor:Int = 1](CalcMethods):
    """
    Containts vectorized functions with unroll_factor = 1 as default, can be changed by user.
    """

    fn __init__(inout self:Self):
        pass

    ##### VECTORIZED FUNCTIONS WITH TWO INPUT TENSORS ######
    fn _mathfunc_2Tinput_1Toutput[T: DType, function: fn[T: DType, simd_wid: Int](self, SIMD[T, simd_wid], SIMD[T, simd_wid]) -> SIMD[T, simd_wid]]
    (self, t1: Tensor[T], t2: Tensor[T]) raises -> Tensor[T]:

        if t1.shape() != t2.shape():
            raise Error("Shapes don't match, cannot apply the given function")

        var result_tensor: Tensor[T] = Tensor[T](t1.shape())
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](t1.data().load[width=simd_width](idx), t2.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts, unroll_factor=unroll_factor](t1.num_elements())
        return result_tensor

    ##### VECTORIZED FUNCTIONS WITH ONE INPUT TENSOR ######
    # Is it possible to parallelize this by splitting the tensor into some N parts?
    fn _mathfunc_1Tinput_1Toutput[T:DType, function: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](
        self, tensor: Tensor[T]) raises -> Tensor[T]:
        
        var result_tensor: Tensor[T] = Tensor[T]()
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](tensor.data().load[width=simd_width](idx)))
        
        vectorize[vectorized, nelts, unroll_factor=unroll_factor](tensor.num_elements())
        return result_tensor

    ##### VECTORIZED FUNCTIONS WITH ONE INPUT TENSOR, ONE INPUT INTEGER #####
    fn _mathfunc_1T1Sinput_1Toutput[T:DType, function: fn[T:DType, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[T, simd_w]](
        self, tensor: Tensor[T], s: Scalar[T]) raises -> Tensor[T]:
        
        var result_tensor: Tensor[T] = Tensor[T]()
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](tensor.data().load[width=simd_width](idx), s))
        
        vectorize[vectorized, nelts, unroll_factor=unroll_factor](tensor.num_elements())
        return result_tensor

    fn _mathfunc_2Tcompare[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t1: Tensor[T], t2: Tensor[T]) raises -> Tensor[DType.Bool]:
        
        if t1.shape() != t2.shape():
            raise Error("Shapes are not compatible")

        var result_tensor: Tensor[DType.Bool] = Tensor[DType.Bool]()
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) raises -> None:
            result_tensor.data().store[width=simd_width](idx, 
            function[T, simd_width](t1.data().load[width=simd_width](idx), t2.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts](t1.num_elements())
        return result_tensor
        
    fn _issomething[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t2: Tensor[T]) raises -> Tensor[DType.Bool]:
        
        var result_tensor: Tensor[DType.Bool] = Tensor[DType.Bool]()
        alias nelts = simdwidthof[T]()
        
        @parameter
        fn vectorized[simd_width: Int](idx: Int) raises -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](t1.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts](t1.num_elements())
        return result_tensor

struct ParallelizedMethod(CalcMethods):
    """
    Containts parallelized functions. 
    """
    fn __init__(self):
        pass

    fn _mathfunc_2Tinput_1Toutput[T: DType, function: fn[T: DType, simd_wid: Int](self, SIMD[T, simd_wid], SIMD[T, simd_wid]) -> SIMD[T, simd_wid]]
    (self, t1: Tensor[T], t2: Tensor[T]) raises -> Tensor[T]:

        if t1.shape() != t2.shape():
            raise Error("Shapes don't match, cannot apply the given function")

        var result_tensor: Tensor[T] = Tensor[T](t1.shape())
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](t1.data().load[width=simd_width](idx), t2.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts, unroll_factor=unroll_factor](t1.num_elements())
        return result_tensor

    fn _mathfunc_1Tinput_1Toutput[T:DType, function: fn[type:DType, simd_w:Int](SIMD[type, simd_w]) -> SIMD[type, simd_w]](
        self, tensor: Tensor[T]) raises -> Tensor[T]:
        
        var result_tensor: Tensor[T] = Tensor[T]()
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](tensor.data().load[width=simd_width](idx)))
        
        vectorize[vectorized, nelts, unroll_factor=unroll_factor](tensor.num_elements())
        return result_tensor

    fn _mathfunc_1T1Sinput_1Toutput[T:DType, function: fn[T:DType, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[T, simd_w]](
        self, tensor: Tensor[T], s: Scalar[T]) raises -> Tensor[T]:
        
        var result_tensor: Tensor[T] = Tensor[T]()
        alias nelts = simdwidthof[T]()

        @parameter
        fn vectorized[simd_width: Int](idx: Int) -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](tensor.data().load[width=simd_width](idx), s))
        
        vectorize[vectorized, nelts, unroll_factor=unroll_factor](tensor.num_elements())
        return result_tensor

    fn _mathfunc_2Tcompare[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w], SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t1: Tensor[T], t2: Tensor[T]) raises -> Tensor[DType.Bool]:
        
        if t1.shape() != t2.shape():
            raise Error("Shapes are not compatible")

        var result_tensor: Tensor[DType.Bool] = Tensor[DType.Bool]()
        alias nelts = simdwidthof[T]()
        
        @parameter
        fn vectorized[simd_width: Int](idx: Int) raises -> None:
            result_tensor.data().store[width=simd_width](idx, 
            function[T, simd_width](t1.data().load[width=simd_width](idx), t2.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts](t1.num_elements())
        return result_tensor
        
    fn _issomething[T:DType, function: fn[T:Dtype, simd_w:Int](SIMD[T, simd_w]) -> SIMD[DType.Bool, simd_w]](
        self, t2: Tensor[T]) raises -> Tensor[DType.Bool]:
        
        var result_tensor: Tensor[DType.Bool] = Tensor[DType.Bool]()
        alias nelts = simdwidthof[T]()

            
        @parameter
        fn vectorized[simd_width: Int](idx: Int) raises -> None:
            result_tensor.data().store[width=simd_width](idx, function[T, simd_width](t1.data().load[width=simd_width](idx)))

        vectorize[vectorized, nelts](t1.num_elements())
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