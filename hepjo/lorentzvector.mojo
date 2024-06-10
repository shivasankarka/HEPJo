# Defaults
from builtin.dtype import DType
from collections.vector import InlinedFixedVector
from algorithm import vectorize
import math

# Modules
from .vector import Vector3D
from .constants import pi

struct LorentzVector[dtype: DType = DType.float64](
    Intable, CollectionElement, Sized
    ):
    var _ptr: DTypePointer[dtype]
    var _size: Int

    fn __init__(inout self, *data:Scalar[dtype]):
        self._size = 4
        self._ptr = DTypePointer[dtype].alloc(self._size)
        memset_zero(self._ptr, self._size)
        for i in range(self._size):
            self._ptr[i] = data[i]

    fn __init__(inout self, inout vector3d:Vector3D[dtype], t:Scalar[dtype]) raises:
        self._size = 4
        self._ptr = DTypePointer[dtype].alloc(self._size)
        self._ptr[0] = vector3d[0]
        self._ptr[1] = vector3d[1]
        self._ptr[2] = vector3d[2]
        self._ptr[3] = t

    fn __str__(self) -> String:
        return "("+str(self._ptr[0])+", "+str(self._ptr[1])+", "+str(self._ptr[2])+", "+str(self._ptr[3])+")"

    fn __copyinit__(inout self, new: Self):
        self._size = new._size
        self._ptr = new._ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._size = existing._size
        self._ptr = existing._ptr
        existing._size = 0
        existing._ptr = DTypePointer[dtype]()

    fn __getitem__(inout self, index:Int) -> Scalar[dtype]:
        return self._ptr.load[width=1](index)

    fn __setitem__(inout self, index:Int, value:Scalar[dtype]):
        self._ptr.store[width=1](index, value)

    fn __del__(owned self):
        self._ptr.free()

    fn __len__(self) -> Int:
        return self._size

    fn __int__(self) -> Int:
        return self._size

    # fn __str__(inout self) -> String:
    #     var printStr:String = "["
    #     var prec:Int=4
    #     for i in range(self._size):
    #         var val = self[i]
    #         @parameter
    #         if _mlirtype_is_eq[Scalar[dtype], Float64]():
    #             var s: String = ""
    #             var int_str: String
    #             int_str = String(trunc(val).cast[DType.int32]())
    #             if val < 0.0:
    #                 val = -val
    #             var float_str: String
    #             if math.mod(val,1)==0:
    #                 float_str = "0"
    #             else:
    #                 float_str = String(mod(val,1))[2:prec+2]
    #             s = int_str+"."+float_str
    #             if i==0:
    #                 printStr+=s
    #             else:
    #                 printStr+="  "+s
    #         else:
    #             if i==0:
    #                 printStr+=str(val)
    #             else:
    #                 printStr+="  "+str(val)

    #     printStr+="]\n"
    #     printStr+="Length:"+str(self._size)+","+" DType:"+str(dtype)
    #     return printStr

    fn __repr__(inout self) -> String:
        return "LorentzVector(x="+str(self._ptr[0])+", y="+str(self._ptr[1])+", z="+str(self._ptr[2])+", t="+str(self._ptr[3])+")"

    # TODO: Implement iterator for Vector3D, I am not sure how to end the loop in __next__ method.
    # fn __iter__(inout self) -> Self:
    #     self.index = -1
    #     return self

    # fn __next__(inout self) -> Scalar[dtype]:
    #     self.index += 1
    #     if self.index == self._size:
    #         # return Optional[Scalar[dtype]]()
    #         return Scalar[dtype]()
    #     else:
    #         return self._ptr[self.index]

    fn __pos__(inout self) -> Self:
        return self*(1.0)

    fn __neg__(inout self) -> Self:
        return self*(-1.0)

    fn __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr

    # ARITHMETICS
    fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self._size)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._ptr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self._size)
        return new_array

    fn _elementwise_array_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._ptr.store[width=simd_width](index, func[dtype, simd_width](self._ptr.load[width=simd_width](index), other._ptr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._size)
        return new_vec

    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[add](other)

    fn __add__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[add](other)

    fn __radd__(inout self, s: Scalar[dtype])->Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn _sub__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[sub](other)

    fn __sub__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[sub](other)

    # TODO: I don't know why I am getting error here, so do this later.
    # fn __rsub__(inout self, s: Scalar[dtype])->Self:
    #     return -(self - s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self-s

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elementwise_scalar_arithmetic[mul](s)

    fn __mul__(self, other: Self)->Self:
        return self._elementwise_array_arithmetic[mul](other)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self*s

    fn _reduce_sum(self) -> Scalar[dtype]:
        var reduced = Scalar[dtype](0.0)
        alias simd_width: Int = simdwidthof[dtype]()
        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced[0] += self._ptr.load[width = simd_width](idx).reduce_add()
        vectorize[vectorize_reduce, simd_width](self._size)
        return reduced

    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[mul](other)._reduce_sum()

    fn __pow__(self, p: Int)->Self:
        return self._elementwise_pow(p)

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = Self(self._size)
        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec._ptr.store[width=simd_width](idx, math.pow(self._ptr.load[width=simd_width](idx), p))
        vectorize[tensor_scalar_vectorize, simd_width](self._size)
        return new_vec

    fn __truediv__(inout self, s: Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[div](s)

    fn __truediv__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[div](other)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other:Self):
        self = self.__truediv__(other)

    fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)


    # PROPERTIES
    fn x(self) -> Scalar[dtype]:
        return self._ptr[0]

    fn y(self) -> Scalar[dtype]:
        return self._ptr[1]

    fn z(self) -> Scalar[dtype]:
        return self._ptr[2]

    fn t(self) -> Scalar[dtype]:
        return self._ptr[3]

    fn vector(self) -> Vector3D[dtype]:
        return Vector3D[dtype](self._ptr[0], self._ptr[1], self._ptr[2])

    fn x(self, value: Scalar[dtype]):
        self._ptr[0] = value

    fn y(self, value: Scalar[dtype]):
        self._ptr[1] = value

    fn z(self, value: Scalar[dtype]):
        self._ptr[2] = value

    fn t(self, value: Scalar[dtype]):
        self._ptr[3] = value

    fn mag(inout self) -> Scalar[dtype]:
        return sqrt(self.x()**2 + self.y()**2 + self.z()**2)

    fn mag2(inout self) -> Scalar[dtype]:
        return self.x()**2 + self.y()**2 + self.z()**2

    fn magl2(inout self) -> Scalar[dtype]:
        return self._ptr[3]**2 - self.mag2()

    fn magl(inout self) -> Scalar[dtype]:
        var magn2 = self.magl2()
        return sqrt(magn2) if magn2 > 0.0 else -sqrt(-magn2)

    fn costheta(inout self) -> Scalar[dtype]:
        if self.mag() == 0.0:
            return 1.0
        else:
            return self._ptr[2]/self.mag()

    fn theta(inout self, degree:Bool=False) -> Scalar[dtype]:
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / Scalar[dtype](pi)
        else:
            return theta

    fn phi(inout self, degree:Bool=False) -> Scalar[dtype]:
        var phi = atan2(self._ptr[1], self._ptr[0])
        if degree == True:
            return phi * 180 / Scalar[dtype](pi)
        else:
            return phi

    fn px(inout self) -> Scalar[dtype]:
        return self.x()

    fn px(inout self, px: Scalar[dtype]):
        self.x(px)

    fn py(inout self) -> Scalar[dtype]:
        return self.y()

    fn py(inout self, py: Scalar[dtype]):
        self.y(py)

    fn pz(inout self) -> Scalar[dtype]:
        return self.z()

    fn pz(inout self, pz: Scalar[dtype]):
        self.z(pz)

    fn e(inout self) -> Scalar[dtype]:
        return self.t()

    fn e(inout self, e: Scalar[dtype]):
        self.t(e)

    fn m(inout self) -> Scalar[dtype]:
        return sqrt(self.e()**2 - self.mag2())

    fn set(inout self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype], t: Scalar[dtype]):
        self.x(x)
        self.y(y)
        self.z(z)
        self.t(t)

    fn setpxpypzm(inout self, px: Scalar[dtype], py: Scalar[dtype], pz: Scalar[dtype], m: Scalar[dtype]):
        self._ptr[0] = px
        self._ptr[1] = py
        self._ptr[2] = pz

        if m > 0.0:
            self._ptr[3] = sqrt(px**2+py**2+pz**2+m**2)
        else:
            self._ptr[3] = sqrt(px**2+py**2+pz**2-m**2)

    fn setpxpypze(inout self, px: Scalar[dtype], py: Scalar[dtype], pz: Scalar[dtype], e: Scalar[dtype]):
        self.set(px, py, pz, e)

    fn setptetaphim(inout self, pt:Scalar[dtype], eta:Scalar[dtype], phi:Scalar[dtype], m:Scalar[dtype]):
        var px = pt*cos(phi)
        var py = pt*sin(phi)
        var pz = pt*sinh(eta)
        self.setpxpypzm(px, py, pz, m)

    fn setptetaphie(inout self, pt:Scalar[dtype], eta:Scalar[dtype], phi:Scalar[dtype], e:Scalar[dtype]):
        var px = pt*cos(phi)
        var py = pt*sin(phi)
        var pz = pt*sinh(eta)
        self.setpxpypze(px, py, pz, e)

    fn tolist(inout self) -> List[Scalar[dtype]]:
        return List[Scalar[dtype]](self._ptr[0],self._ptr[1],self._ptr[2],self._ptr[3])

    fn p(inout self) -> Scalar[dtype]:
        return self.mag()

    fn perp(inout self) -> Scalar[dtype]:
        return sqrt(self._ptr[0]**2 + self._ptr[1]**2)

    fn pt(inout self) -> Scalar[dtype]:
        return self.perp()

    fn et(inout self) -> Scalar[dtype]:
        return self.e() * (self.pt() / self.p())

    fn minv(inout self) -> Scalar[dtype]:
        return self.magl()

    fn mt(inout self) -> Scalar[dtype]:
        return sqrt(self.mt2()) 

    fn mt2(inout self) -> Scalar[dtype]:
        return self.e()**2 - self.pz()**2
    
    fn beta(inout self) -> Scalar[dtype]:
        return self.p() / self.e()

    fn gamma(inout self) -> Scalar[dtype]:
        if self.beta() < 1:
            return 1.0 / sqrt(1.0 - self.beta()**2)
        else:
            print("Gamma > 1.0, Returning 10e10")
            return 10e10

    fn eta(inout self) -> Scalar[dtype]:
        if abs(self.costheta()) < 1.0:
            return -0.5 * log((1.0 - self.costheta()) / (1.0 + self.costheta()))  
        else:
            print("eta > 1.0, Returning 10e10")
            return 10e10 if self.z() > 0 else -10e10

    fn pseudorapidity(inout self) -> Scalar[dtype]:
       return self.eta()

    fn rapidity(inout self) -> Scalar[dtype]:
        return 0.5 * log((self.e() + self.pz()) / (self.e() - self.pz()))

    fn copy(inout self) -> Self:
        return Self(self._ptr[0], self._ptr[1], self._ptr[2], self._ptr[3])

    # Implement iter
    fn boostvector(inout self) -> Vector3D[dtype]:
        return Vector3D(self.x()/self.t(), self.y()/self.t(), self.z()/self.t())

    fn boost(inout self, inout args:Vector3D[dtype]) raises -> Self:  
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx:Scalar[dtype] = args[0]
        var by:Scalar[dtype] = args[1]
        var bz:Scalar[dtype] = args[2]

        var b2:Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma:Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp:Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2:Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp:Scalar[dtype] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp:Scalar[dtype] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp:Scalar[dtype] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    fn boostplus(inout self, inout args:Vector3D[dtype]) raises -> Self:  
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx:Scalar[dtype] = args[0]
        var by:Scalar[dtype] = args[1]
        var bz:Scalar[dtype] = args[2]

        var b2:Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma:Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp:Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2:Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp:Scalar[dtype] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp:Scalar[dtype] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp:Scalar[dtype] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    fn boostminus(inout self, inout args:Vector3D[dtype]) raises -> Self:  
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx:Scalar[dtype] = -1.0 *args[0]
        var by:Scalar[dtype] = -1.0 *args[1]
        var bz:Scalar[dtype] = -1.0 *args[2]

        var b2:Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma:Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp:Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2:Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp:Scalar[dtype] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp:Scalar[dtype] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp:Scalar[dtype] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    # maybe you can change this implementation
    fn dot(inout self, other:Self) -> Scalar[dtype]:
        return self.t() * other.t() - self._elementwise_array_arithmetic[mul](other)._reduce_sum()

    fn isspacelike(inout self) -> Bool:
        if self.magl2() != 0.0:
            return self.magl2() < 0.0 
        else:
            print("Magnitude is zero, returning False")
            return False

    fn istimelike(inout self) -> Bool:
        if self.magl2() != 0.0:
            return self.magl2() > 0.0
        else:
            print("Magnitude is zero, returning False")
            return False
    
    fn islightlike(inout self) -> Bool:
        return self.magl2() == 0.0

    fn torestframe(inout self) raises -> Self:
        var boost_vec = self.boostvector()
        return self.boostplus(boost_vec)
    




