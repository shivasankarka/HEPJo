import math as mt

import hepjo as hj

fn main() raises:
    # * Vector3D
    var v3d = hj.Vector3D[DType.float32].frompoint(1.0, 2.0, 3.0)
    print("v3d: ", v3d) #prints [1.0  2.0  3.0], dtype=float32, Length=3
    print(3.0 - v3d) #prints [2.0  1.0  0.0], dtype=float32, Length=3
    print(v3d - 3.0) #prints [-2.0  -1.0  -0.0], dtype=float32, Length=3
    var v3d_2 = hj.Vector3D.fromvector(v3d) # creates a new 3d vector from v3d
    print(v3d * v3d_2) # prints 
    var v_tensor = v3d.to_tensor()
    print(v_tensor)# prints Tensor([[1.0, 2.0, 3.0]], dtype=float32, shape=3)
    
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


