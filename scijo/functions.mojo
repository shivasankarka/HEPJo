from scijo.vector_pure_traits import Vector3D

# TODO: Vectorize this function
fn trapz(inout x: Vector3D[DType.float64], inout y: Vector3D[DType.float64]) -> Float64:
    var integral: Float64 = 0.0
    for i in range(x.__len__() - 1):
        integral += (x[i] + x[i + 1]) * (y[i] + y[i + 1]) / 2.0
    return integral

