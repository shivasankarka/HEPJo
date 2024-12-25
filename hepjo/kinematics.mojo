from .constants import hbar
from .vector import LorentzVector


# Cross Section Helpers
fn mandelstam_s[
    dtype: DType = DType.float64
](p1: LorentzVector[dtype], p2: LorentzVector[dtype]) -> Scalar[dtype]:
    """
    Calculate Mandelstam s variable.
    """
    return (p1 + p2).mass2()


fn mandelstam_t[
    dtype: DType = DType.float64
](p1: LorentzVector[dtype], p2: LorentzVector[dtype]) -> Scalar[dtype]:
    """
    Calculate Mandelstam t variable.
    """
    return (p1 - p2).mass2()


# ! I should change these after implmenting unit system.
def cross_section_to_events(cs: Float64, lumi: Float64) -> Float64:
    """
    Convert cross section (pb) to number of events given luminosity (fb^-1).
    """
    return cs * lumi * 1000  # pb * fb^-1 * (1000 pb/fb)


fn Kallen_lambda[
    dtype: DType = DType.float64
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


fn lifetime_to_width[
    dtype: DType = DType.float64
](tau: Scalar[dtype]) -> Scalar[dtype]:
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


fn width_to_lifetime[
    dtype: DType = DType.float64
](Gamma: Scalar[dtype]) -> Scalar[dtype]:
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
