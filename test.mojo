import scijo as sj
from tensor import Tensor, TensorShape
from scijo.interpolate import interp1d
import math as mt

fn main() raises:

    # * Vector2D
    # var v1_2d = Vector2D[DType.float32].frompoint(1.0, 3.0)
    # var v2_2d = Vector2D.fromvector(v1_2d)
    # var v3_mul = v1_2d * v2_2d
    # print(v3_mul)
    # print(v1_2d@v2_2d)

    # * Vector3D
    # var v1_3d = Vector3D[DType.float32].frompoint(1.0, 2.0, 3.0)
    # var v2_3d = Vector3D.fromvector(v1_3d)
    # var v3_mul = v1_3d * v2_3d
    # print(v3_mul)
    # print(v1_3d@v2_3d)
    # var v_tensor = v1_3d.to_tensor()
    # print(v_tensor)

    # arrays
    var arr = sj.array[DType.float32](TensorShape(3), List[Float32](1.0,2.0,4.0))
    var arr1 = sj.array[DType.float32](arr)
    # var diff = arr1 + arr
    print(arr1)
    # var x = Tensor[DType.float64](TensorShape(5), List[Float64](1.0,2.0,3.0,4.0,5.0))
    # var y = Tensor[DType.float64](TensorShape(5), List[Float64](10.0,3.0,50.0,1.0,15.0))
    # var xint = Tensor[DType.float64](TensorShape(5), List[Float64](3.0,4.0,5.0,6.0,7.0))
    # var arr3 = interp1d(xint, x, y, method="linear", fill_value="extrapolate")
    # var unsorted = Tensor[DType.float64](TensorShape(5), List[Float64](12.0,2.0,3.0,4.0,5.0))
    # print(arr3)
    # print(mean(x))
    # print(max(x))
    # print(min(x))
    # print(binary_sort(unsorted))
    # print(median(unsorted))
    # print(mode(unsorted))
    # print("var: ", variance(x))
    # print("var: ", variance(x, mu=mean(x)))

    # var arr_linear = sj.linspace[DType.float32](start=1.0, stop=10.0, num=10, endpoint=True)
    # var arr_linearp = sj.linspace[DType.float32](start=1.0, stop=10.0, num=10, endpoint=True, parallel=True)
    # var arr_log = sj.logspace[DType.float32](start=-2.0, stop=0.0, num=10, endpoint=False)
    # var arr_logp = sj.logspace[DType.float32](start=-2.0, stop=0.0, num=10, endpoint=False, parallel=True)

    # var arr1 = Tensor[DType.float64](TensorShape(3, 3))
    # print(arr_linear.__str__())
    # print(arr_linearp.__str__())
    # var min_val = sj.minimum[DType.float32](arr_linear, arr_log)
    # print(arr_linear)
    # print(arr_linear.data().load[width=2](0).reduce[math.sub[DType.float32]]())
    # print(sj.trapz(arr_linear, arr_linearp))
    # print(arr_linear.data().__str__())
    # print(arr_linear.data().simd_strided_load[width=2](2))
    # print(arr_linear.data().load[width=2](1))
    # print(sj.pInf[DType.float32]())
    # print(sj.mInf[DType.float32]())
    # if sj.pInf[DType.float32]() < 5:
    #     print("true")
    # else:
    #     print("false")
    # print(sj.max[DType.float64](y))
    # print(sj.max_vec[DType.float64](y))
    # print(sj.min[DType.float64](y))
    # print(sj.argmax[DType.float64](y))
    # print(sj.argmin[DType.float64](y))
    # var pow2 = sj.power(arr_linear, 2.0)
    # print(pow2)
    # print(arr.load[width=3](0,1))
    
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

    # var empty_arr = sj.zeros[DType.float32](2,5)
    # print(empty_arr)
    # var eye_arr = sj.eye[DType.float32](3,3)
    # print(eye_arr)
    # var one_arr = sj.ones[DType.float32](2,4)
    # print(one_arr)
    # var fill_arr = sj.fill[DType.float32, 3.0](3,2)
    # print(fill_arr)
    # var fill_arr1 = sj.fill[DType.float32](VariadicList[Int](2,2), 5.0)
    # print(fill_arr1)
    # var geospace_arr = sj.geomspace[DType.float32](1, 256, num=9, endpoint=True)
    # print(geospace_arr)


