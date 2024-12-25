fn is_close[
    dtype: DType = DType.float64
](
    a: Scalar[dtype],
    b: Scalar[dtype],
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
) -> Bool:
    """
    Check if two scalars are close to each other.

    Parameters:
            dtype: Data type of the input and output arrays.

    Arguments:
            a: Scalar.
            b: Scalar.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

    Returns:
            True if the two scalars are close to each other, False otherwise.
    """
    return abs(a - b) <= atol + rtol * abs(b)
