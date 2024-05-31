

fn main() raises:

    # * Vector2D
    var v1_2d = Vector2D[DType.float32].frompoint(1.0, 3.0)
    var v2_2d = Vector2D.fromvector(v1_2d)
    var v3_mul = v1_2d * v2_2d
    print(v3_mul)
    print(v1_2d@v2_2d)

    # print(sj.const[DType.float32].e)
    # var v4 = sj.sin[DType.float32](v1)
    # print(v4)

    # var arr3 = LorentzVector(1.0, 2.0, 3.0, 10.0)
    # var boostvec = arr3.boostvector()
    # print(arr3.__repr__())
    # print(arr3.minv())
    # print(arr3.isspacelike())
    # arr3 = arr3.torestframe()
    # print(arr3.__repr__())

    # var mat = sj.Vector3D(3, 3, 3)
    # var mat1 = sj.Vector3D(1, 2, 3)

    # # print(mat.__str__())
    # var arr = sj.array(3, 3)
    # arr = arr / 2.0
    # # print(arr)

    # var a = sj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # var b = sj.Vector3D(List[Float32](1.0, 2.0, 3.0))
    # print(a.__str__())
    # print(b.__str__())
    # var c = a + b
    # print(c.__str__())

    # assert_equal[SIMD[DType.float32]](1.0, 1.0, msg="true")
    # var a = Tensor[DType.float64](TensorShape(3), List[Float64](1.0,2.0,3.0))
    # var a = Tensor[DType.float64](3)
    # a[0] = 1.0
    # a[1] = 2.0
    # a[2] = 3.0
    # var v = sin(a) 
