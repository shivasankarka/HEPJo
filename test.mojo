import math as mt
import time
from algorithm import parallelize
from benchmark.compiler import keep

import hepjo as hj
from random import random_float64

# fn main() raises:
#     # * Vector3D
#     var v3d = hj.Vector3D[DType.float32].frompoint(1.0, 2.0, 3.0)
#     print("v3d: ", v3d) #prints [1.0  2.0  3.0], dtype=float32, Length=3
#     print(3.0 - v3d) #prints [2.0  1.0  0.0], dtype=float32, Length=3
#     print(v3d - 3.0) #prints [-2.0  -1.0  -0.0], dtype=float32, Length=3
#     var v3d_2 = hj.Vector3D.fromvector(v3d) # creates a new 3d vector from v3d
#     print(v3d * v3d_2) # prints 
#     var v_tensor = v3d.to_tensor()
#     print(v_tensor)# prints Tensor([[1.0, 2.0, 3.0]], dtype=float32, shape=3)
    
    # * Vector2D
    # var v1_2d = Vector2D[DType.float32].frompoint(1.0, 3.0)
    # var v2_2d = Vector2D.fromvector(v1_2d)
    # var v3_mul = v1_2d * v2_2d
    # print(v3_mul)
    # print(v1_2d@v2_2d)

    # * Lorentz Vector
    # var arr3 = LorentzVector(1.0, 2.0, 3.0, 10.0)
    # var boostvec = arr3.boostvector()
    # print(arr3.__repr__())
    # print(arr3.minv())
    # print(arr3.isspacelike())
    # arr3 = arr3.torestframe()
    # print(arr3.__repr__())

    # var mat = hj.Vector3D(3, 3, 3)
    # var mat1 = hj.Vector3D(1, 2, 3)

    # # print(mat.__str__())
    # var arr = hj.array(3, 3)
    # arr = arr / 2.0
    # # print(arr)

    # var a = hj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # var b = hj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # print(a.__str__())
    # print(b.__str__())
    # var c = a + b
    # print(c.__str__())
import os

fn main() raises:
    var t0 = time.now()
    # alias file_name = "/Users/shivasankar/Documents/Research/Codes/FASER/Models/DiracRHN/model/events/lnuq/100MeV/events_lnle_10000.csv"
    # var file = open(file_name, "r")  

    # var line = file.read()
    # var l = line
    # var count = 0
    # for i in range(len(line)):
    #     var l = line[i]
    #     if ord(l) == 10:
    #         var line1 = line[count:i]
    #         count = i
    #         var vals = line1.split(",")
    #         if i > 88:
    #             print(atof(vals[1]))
    #             print(atof(vals[7]), atof(vals[8]), atof(vals[9]), atof(vals[13]))
    #             print(atof(vals[3]), atof(vals[4]), atof(vals[5]))
    #             print()
    #         # print(vals.__str__())

    #     if i > 10000:
    #         break
    #     i += 1
    # file.close()

    @parameter
    fn calc(idx: Int) -> None:
        var r0 = SIMD[DType.float64, 4](-0.77701472595259997, -0.12122280619049999 , 6.00117508534040011, 0.0)
        # var v0 = SIMD[DType.float64, 4](random_float64(1e-5, 1e-4) * 3e8 * (1.0 / 0.511e-3), random_float64(1e-5, 1e-4) * 3e8 * (1.0 / 0.511e-3), 0.99999999 * 3e8 * (1.0 / 0.511e-3), 0.0)
        var v0 = SIMD[DType.float64, 4](( -1.0202484708999999/788.1528091852) * 3e8 * (1.0 / 0.511e-3), (0.17683364600000001/788.1528091852) * 3e8 * (1.0 / 0.511e-3), (788.15212900120002/788.1528091852) * 3e8 * (1.0 / 0.511e-3), 0.0)
         
        var a0 = SIMD[DType.float64, 4](0.0, 0.0, 0.0, 0.0)
        var ratio: Float64 = -175824175824.17584
        var B = SIMD[DType.float64, 4](0, 1.0, 0, 0.0)
        print("Initial position: ", r0.__str__())
        print("Initial velocity: ", v0.__str__())
        print("Initial energy: ", v0[0]**2 + v0[1]**2 + v0[2]**2)

        hj.simulate_particles(r = r0, v = v0, a = a0, B = B, zf = 10.0, num_steps = 10**6, dt = 1e-13, ratio = ratio)

        print("Final position: ", r0[0], r0[1], r0[2])
        print("Final velocity: ", v0[0], v0[1], v0[2])
        print("Final energy: ", v0[0]**2 + v0[1]**2 + v0[2]**2)

    parallelize[calc](100, 100)
    print("time taken: ", (time.now() - t0) / 1e9)