from algorithm.functional import vectorize
from memory import UnsafePointer
from sys import simdwidthof


fn bool_simd_store[
    simd_width: Int
](
    ptr: UnsafePointer[Scalar[DType.bool]],
    start: Int,
    val: SIMD[DType.bool, simd_width],
):
    """
    Work around function for storing bools from a simd into a DTypePointer.

    Parameters:
        simd_width: Number of items to be retrieved.

    Args:
        ptr: Pointer to be retreived from.
        start: Start position in pointer.
        val: Value to store at locations.
    """
    (ptr + start).strided_store(val, 1)


# VECTORIZED MATH OPERATIONS ON VECTOR3D
fn compare_2_vectors[
    size: Int,
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[DType.bool, simd_w],
](
    array1: UnsafePointer[Scalar[dtype]],
    array2: UnsafePointer[Scalar[dtype]],
    result: UnsafePointer[Scalar[DType.bool]],
) raises:
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data1 = array1.load[width=simdwidth](i)
        var simd_data2 = array2.load[width=simdwidth](i)
        bool_simd_store[simdwidth](
            result,
            i,
            func[dtype, simdwidth](simd_data1, simd_data2),
        )

    vectorize[closure, opt_nelts](size)


fn compare_vector_and_scalar[
    size: Int,
    dtype: DType,
    func: fn[type: DType, simd_w: Int] (
        SIMD[type, simd_w], SIMD[type, simd_w]
    ) -> SIMD[DType.bool, simd_w],
](
    array1: UnsafePointer[Scalar[dtype]],
    scalar: SIMD[dtype, 1],
    result: UnsafePointer[Scalar[DType.bool]],
) raises:
    alias opt_nelts = simdwidthof[dtype]()

    @parameter
    fn closure[simdwidth: Int](i: Int):
        var simd_data1 = array1.load[width=simdwidth](i)
        var simd_data2 = SIMD[dtype, simdwidth](scalar)
        bool_simd_store[simdwidth](
            result,
            i,
            func[dtype, simdwidth](simd_data1, simd_data2),
        )

    vectorize[closure, opt_nelts](size)


fn elementwise_scalar_arithmetic[
    size: Int,
    dtype: DType,
    function: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
](
    vector: UnsafePointer[Scalar[dtype]],
    scalar: Scalar[dtype],
    result: UnsafePointer[Scalar[dtype]],
):
    """
    Performs an element-wise scalar arithmetic operation on this vector using SIMD,
    This function applies a specified arithmetic operation to each element of the vector
    in conjunction with a scalar value such as
        self + s
        self - s
        self * s
        self / s.

    Parameters:
        size: Size of the vector.
        dtype: Data type of the elements.
        function: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.

    Args:
        vector: The vector to be added.
        scalar: The scalar value to be used in the function operation.
        result: The vector to store the result of the operation.
    """
    alias opt_nelts: Int = simdwidthof[dtype]()

    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        var simd_data1 = vector.load[width=simd_width](idx)
        var simd_data2 = SIMD[dtype, simd_width](scalar)
        result.store(idx, function[dtype, simd_width](simd_data1, simd_data2))

    vectorize[elemwise_vectorize, opt_nelts](size)


fn elementwise_array_arithmetic[
    size: Int,
    dtype: DType,
    function: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
](
    vector1: UnsafePointer[Scalar[dtype]],
    vector2: UnsafePointer[Scalar[dtype]],
    result: UnsafePointer[Scalar[dtype]],
):
    """
    Performs an element-wise arithmetic operation between two vectors using SIMD (Single Instruction, Multiple Data) techniques.

    This function leverages a provided SIMD-compatible function `func` to perform the specified arithmetic operation on corresponding elements of this vector and another vector `other`.

    Parameters:
        size: The size of the vector.
        dtype: Datatype of the elements.
        function: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.

    Args:
        vector1: The first vector.
        vector2: The second vector.
        result: The vector to store the result of the operation.

    """
    alias opt_nelts: Int = simdwidthof[dtype]()
    # var new_array = UnsafePointer[Scalar[dtype]]()

    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        var simd_data1 = vector1.load[width=simd_width](idx)
        var simd_data2 = vector2.load[width=simd_width](idx)
        result.store(idx, function[dtype, simd_width](simd_data1, simd_data2))

    vectorize[elemwise_vectorize, opt_nelts](size)


fn elementwise_array_arithmetic[
    size: Int,
    dtype: DType,
    function: fn[dtype: DType, width: Int] (
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
](
    vector1: UnsafePointer[Scalar[dtype]],
    vector2: UnsafePointer[Scalar[dtype]],
    mut result: Scalar[dtype],
):
    """
    Performs an element-wise arithmetic operation between two vectors using SIMD (Single Instruction, Multiple Data) techniques.

    This function leverages a provided SIMD-compatible function `func` to perform the specified arithmetic operation on corresponding elements of this vector and another vector `other`.

    Parameters:
        size: The size of the vector.
        dtype: Datatype of the elements.
        function: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.

    Args:
        vector1: The first vector.
        vector2: The second vector.
        result: The vector to store the result of the operation.

    """
    alias opt_nelts: Int = simdwidthof[dtype]()
    # var new_array = UnsafePointer[Scalar[dtype]]()

    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        var simd_data1 = vector1.load[width=simd_width](idx)
        var simd_data2 = vector2.load[width=simd_width](idx)
        result += function[dtype, simd_width](
            simd_data1, simd_data2
        ).reduce_add()

    vectorize[elemwise_vectorize, opt_nelts](size)


fn elementwise_function_arithmetic[
    size: Int,
    dtype: DType,
    func: fn[dtype: DType, width: Int] (SIMD[dtype, width]) -> SIMD[
        dtype, width
    ],
](inout vector: UnsafePointer[Scalar[dtype]]):
    """
    Applies a SIMD-compatible function element-wise to this vector.

    This function takes a SIMD-compatible function `func` that operates on a single SIMD type and applies it to each element of the vector, effectively transforming each element based on the function's logic.

    Parameters:
        size: The size of the vector.
        dtype: Dataype of the elements.
        func: A function that takes a SIMD type and returns a SIMD type, defining the operation to be performed on each element.

    """
    alias opt_nelts: Int = simdwidthof[dtype]()

    @parameter
    fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
        vector.store(
            idx, func[dtype, simd_width](vector.load[width=simd_width](idx))
        )

    vectorize[elemwise_vectorize, opt_nelts](size)
