from .constants import hbar


fn Kallen_lambda[
    dtype: DType = f64
](a: Scalar[dtype], b: Scalar[dtype], c: Scalar[dtype]) -> Scalar[dtype]:
    """
    Kallen lambda function.

    Parameters:
            dtype: Data type of the input and output arrays.

    Arguments:
            a: Scalar.
            b: Scalar.
            c: Scalar.

    Returns:
            Scalar: Result of the Kallen lambda function.
    """
    return (a - b - c) ** 2 - 4 * b * c


fn lifetime_to_width[dtype: DType = f64](tau: Scalar[dtype]) -> Scalar[dtype]:
    """
    Convert lifetime to width.

    Parameters:
            dtype: Data type of the input and output arrays.

    Arguments:
            tau: Scalar.

    Returns:
            Scalar: Result of the conversion.
    """
    return 1.0 / tau


fn width_to_lifetime[dtype: DType = f64](Gamma: Scalar[dtype]) -> Scalar[dtype]:
    """
    Convert width to lifetime.

    Parameters:
            dtype: Data type of the input and output arrays.

    Arguments:
            width: Scalar.

    Returns:
            Scalar: Result of the conversion.
    """
    return 1.0 / Gamma
