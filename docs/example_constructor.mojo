from hepjo import *

fn main() raises:
    var a = vec3(1, 2, 3)
    var b = vec3(2, 3, 4)
    print(a.dot(b))

    var vec = LorentzVector[f64](3.0, 4.0, 0.0, 5.0)
    print(vec.mag())

    var empty = vec2()
    print("empty: ", empty)
    var xy = vec2(x=1, y=2)
    print("xy: ", xy)
    var vardic = vec2(1, 2)
    print("vardic: ", vardic)
    var list = vec2(List(1.0, 2.0))
    print("list: ", list)
    var simd2 = vec2(SIMD[f64, 1](2))
    print("simd2: ", simd2)
    print(xy.dot(vardic))
