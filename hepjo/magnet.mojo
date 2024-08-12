from algorithm import vectorize
from benchmark.compiler import keep


fn simulate_particles(
    inout r: SIMD[DType.float64, 4],
    inout v: SIMD[DType.float64, 4],
    inout a: SIMD[DType.float64, 4],
    B: SIMD[DType.float64, 4],
    zf: Float64,
    num_steps: Int,
    dt: Float64,
    ratio: Float64,
) -> None:
    alias nelts = simdwidthof[DType.float64]()
    while r[2] <= zf:
        # Calculate k1
        var a_k1 = SIMD[DType.float64, 4](
            (ratio) * (v[1] * B[2] - v[2] * B[1]),
            (ratio) * (v[2] * B[0] - v[0] * B[2]),
            (ratio) * (v[0] * B[1] - v[1] * B[0]),
            0.0,
        )
        var v_k1 = v + a_k1 * dt
        var r_k1 = r + v_k1 * dt / 2

        # Calculate k2
        var a_k2 = SIMD[DType.float64, 4](
            (ratio) * (v_k1[1] * B[2] - v_k1[2] * B[1]),
            (ratio) * (v_k1[2] * B[0] - v_k1[0] * B[2]),
            (ratio) * (v_k1[0] * B[1] - v_k1[1] * B[0]),
            0.0,
        )
        var v_k2 = v + a_k2 * dt / 2
        var r_k2 = r + v_k2 * dt / 2

        # Calculate k3
        var a_k3 = SIMD[DType.float64, 4](
            (ratio) * (v_k2[1] * B[2] - v_k2[2] * B[1]),
            (ratio) * (v_k2[2] * B[0] - v_k2[0] * B[2]),
            (ratio) * (v_k2[0] * B[1] - v_k2[1] * B[0]),
            0.0,
        )
        var v_k3 = v + a_k3 * dt / 2
        var r_k3 = r + v_k3 * dt / 2

        # Calculate k4
        var a_k4 = SIMD[DType.float64, 4](
            (ratio) * (v_k3[1] * B[2] - v_k3[2] * B[1]),
            (ratio) * (v_k3[2] * B[0] - v_k3[0] * B[2]),
            (ratio) * (v_k3[0] * B[1] - v_k3[1] * B[0]),
            0.0,
        )
        var v_k4 = v + a_k4 * dt
        var r_k4 = r + v_k4 * dt

        # Update v and r using weighted average
        v += (a_k1 + 2 * a_k2 + 2 * a_k3 + a_k4) * (dt / 6)
        r += (r_k1 + 2 * r_k2 + 2 * r_k3 + r_k4) * (dt / 6)

        if abs(r[0]) >= 3.0 or abs(r[1]) >= 1.0:
            break
