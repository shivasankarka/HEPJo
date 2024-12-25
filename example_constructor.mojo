from hepjo import *


fn main() raises:
    var a = v3d(1, 2, 3)
    var b = v3d(2, 3, 4)
    print(a.dot(b))

    var vec = LorentzVector[f64](3.0, 4.0, 0.0, 5.0)
    print(vec.mag())

    var empty = v2d()
    print("empty: ", empty)
    var xy = v2d(x=1, y=2)
    print("xy: ", xy)
    var vardic = v2d(1, 2)
    print("vardic: ", vardic)
    var list = v2d(List(1.0, 2.0))
    print("list: ", list)
    var simd2 = v2d(SIMD[f64, 1](2))
    print("simd2: ", simd2)
    print(xy.dot(vardic))
