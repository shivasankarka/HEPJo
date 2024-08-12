from .vector import DTypePointer, Vector2D
from algorithm.functional import vectorize
from .traits import vectors

"""
TODO:
1) Rewrite the docstrings correctly. 
"""

fn bool_simd_store[
    width: Int
](ptr: DTypePointer[DType.bool], start: Int, val: SIMD[DType.bool, width]):
    """
    Work around function for storing bools from a simd into a DTypePointer.

    Parameters:
        width: Number of items to be retrieved.
    
    Args:
        ptr: Pointer to be retreived from.
        start: Start position in pointer.
        val: Value to store at locations.
    """
    (ptr + start).simd_strided_store[width=width, T=Int](val, 1)

# VECTORIZED MATH OPERATIONS ON VECTOR3D
fn compare_2_vectors[size: Int, 
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        array1: DTypePointer[dtype], array2: DTypePointer[dtype] 
    ) raises -> DTypePointer[DType.bool]:
        var result_array: DTypePointer[DType.bool] = DTypePointer[DType.bool]()
        alias opt_nelts = simdwidthof[dtype]()
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = array2.load[width=simdwidth](i)
            bool_simd_store[simdwidth](
                result_array,
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, opt_nelts](size)
        return result_array

fn compare_vector_and_scalar[
        size: Int,
        dtype: DType,
        func: fn[type: DType, simd_w: Int] (
            SIMD[type, simd_w], SIMD[type, simd_w]
        ) -> SIMD[DType.bool, simd_w],
    ](
        array1: DTypePointer[dtype], scalar: SIMD[dtype, 1]
    ) raises -> DTypePointer[DType.bool]:
        var result_array: DTypePointer[DType.bool] = DTypePointer[DType.bool]()
        alias opt_nelts = simdwidthof[dtype]()
        @parameter
        fn closure[simdwidth: Int](i: Int):
            var simd_data1 = array1.load[width=simdwidth](i)
            var simd_data2 = SIMD[dtype,simdwidth].splat(scalar) 
            bool_simd_store[simdwidth](
                result_array,
                i,
                func[dtype, simdwidth](simd_data1, simd_data2),
            )

        vectorize[closure, opt_nelts](size)
        return result_array

fn elementwise_scalar_arithmetic[size: Int, dtype: DType, function: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](vector: DTypePointer[dtype], scalar: Scalar[dtype]) -> DTypePointer[dtype]:
    """
    Performs an element-wise scalar arithmetic operation on this vector using SIMD. 
    This function applies a specified arithmetic operation to each element of the vector 
    in conjunction with a scalar value such as 
        self + s
        self - s
        self * s
        self / s

    Parameters:
        size: Size of the vector.
        dtype: Data type of the elements. 
        function: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
    
    Args:
        vector: The vector to be added. 
        scalar: The scalar value to be used in the function operation. 

    Returns:
        A new instance of the vector where each element is the result of applying the arithmetic operation between the scalar `s` and the corresponding element of the original vector.
    """
    alias opt_nelts: Int = simdwidthof[dtype]()
    var new_array = DTypePointer[dtype]()
    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        var simd_data1 = vector.load[width= simd_width](idx)
        var simd_data2 = SIMD[dtype, simd_width].splat(scalar) 
        new_array.store[width=simd_width](idx, function[dtype, simd_width](simd_data1, simd_data2))
    vectorize[elemwise_vectorize, opt_nelts](size)
    return new_array

fn elementwise_array_arithmetic[size: Int, dtype: DType, function: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](vector1: DTypePointer[dtype], vector2: DTypePointer[dtype]) -> DTypePointer[dtype]:
    """
    Performs an element-wise arithmetic operation between two vectors using SIMD (Single Instruction, Multiple Data) techniques.
    
    This function leverages a provided SIMD-compatible function `func` to perform the specified arithmetic operation on corresponding elements of this vector and another vector `other`.
    
    Parameters:
        size: The size of the vector        
        dtype: Datatype of the elements.
        function: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
    
    Args:
        vector1: The first vector.
        vector2: The second vector. 
    
    Returns:
        A new vector instance where each element is the result of the arithmetic operation performed on corresponding elements of the two input vectors.
    """
    alias opt_nelts: Int = simdwidthof[dtype]()
    var new_array = DTypePointer[dtype]()
    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        var simd_data1 = vector1.load[width= simd_width](idx)
        var simd_data2 = vector2.load[width= simd_width](idx)
        new_array.store[width=simd_width](idx, function[dtype, simd_width](simd_data1, simd_data2))
    vectorize[elemwise_vectorize, opt_nelts](size)
    return new_array

fn elementwise_function_arithmetic[size: Int, dtype: DType, func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](inout vector: DTypePointer[dtype]):
    """
    Applies a SIMD-compatible function element-wise to this vector.
    
    This function takes a SIMD-compatible function `func` that operates on a single SIMD type and applies it to each element of the vector, effectively transforming each element based on the function's logic.
    
    Parameters:
        size: The size of the vector        
        dtype: Dataype of the elements. 
        func: A function that takes a SIMD type and returns a SIMD type, defining the operation to be performed on each element.
    
    """
    alias opt_nelts: Int = simdwidthof[dtype]()
    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        vector.store[width=simd_width](idx, func[dtype, simd_width](vector.load[width=simd_width](idx)))
    vectorize[elemwise_vectorize, opt_nelts](size)