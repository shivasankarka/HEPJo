from builtin.dtype import DType
from sys.intrinsics import _mlirtype_is_eq
from algorithm.functional import elementwise
from algorithm import vectorize

from math import add, sub, mul, div, sin, cos, sqrt, acos, atan2, mod, trunc
from .constants import pi

# alias type = DType.float32
# alias type1 = Float32

# TODO: removed 'Stringable' trait for now since it seems to not recognize __str__ method for some reason.
struct Vector3D[dtype: DType = DType.float64](
    Intable, CollectionElement, Sized
    ):
    var _ptr: DTypePointer[dtype]
    var _size: Int

    # Constructors
    # * I need to figure out how to assign the datatype given by user if possible
    fn __init__(inout self):
        # default constructor
        self._size = 3
        self._ptr =  DTypePointer[dtype].alloc(self._size)
        memset_zero(self._ptr, self._size)

    fn __init__(inout self, *data:Scalar[dtype]):
        self._size = 3

        self._ptr = DTypePointer[dtype].alloc(self._size)
        for i in range(self._size):
            self._ptr[i] = data[i]

    fn __init__(inout self, _ptr: DTypePointer[dtype]):
        self._size = 3
        self._ptr = _ptr

    fn __copyinit__(inout self, new: Self):
        self._size = new._size
        self._ptr = new._ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._size = existing._size
        self._ptr = existing._ptr
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

    fn __str__(inout self) -> String:
        var printStr:String = "["
        var prec:Int=4
        for i in range(self._size):
            var val = self[i]
            @parameter
            if _mlirtype_is_eq[Scalar[dtype], Float64]():
                var s: String = ""
                var int_str: String
                int_str = String(trunc(val).cast[DType.int32]())
                if val < 0.0:
                    val = -val
                var float_str: String
                if math.mod(val,1)==0:
                    float_str = "0"
                else:
                    float_str = String(mod(val,1))[2:prec+2]
                s = int_str+"."+float_str
                if i==0:
                    printStr+=s
                else:
                    printStr+="  "+s
            else:
                if i==0:
                    printStr+=str(val)
                else:
                    printStr+="  "+str(val)

        printStr+="]\n"
        printStr+="Length:"+str(self._size)+","+" DType:"+str(dtype)
        return printStr

    fn __repr__(inout self) -> String:
        return "Vector3D({self.__value[0]}, {self.__value[1]}, {self.__value[2]})"

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

    fn __matmul__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[mul](other)

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

    @staticmethod
    fn origin(inout self) -> Self:
        # Class method to create an origin vector
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(inout self, x:Scalar[dtype], y:Scalar[dtype], z:Scalar[dtype]) -> Self:
        return Self(x, y, z)

    @staticmethod
    fn fromvector(inout v:Self) -> Self:
        return Self(v[0], v[1], v[2])

    @staticmethod
    fn fromsphericalcoords(r:Scalar[dtype], theta:Scalar[dtype], phi:Scalar[dtype]) -> Self:
        var x:Scalar[dtype] = r * sin(theta) * cos(phi)
        var y:Scalar[dtype] = r * sin(theta) * sin(phi)
        var z:Scalar[dtype] = r * cos(theta)
        return Vector3D(x,y,z)

    @staticmethod
    fn fromcylindricalcoodinates(rho:Scalar[dtype], phi:Scalar[dtype], z:Scalar[dtype]) -> Self:
        var x:Scalar[dtype] = rho * cos(phi)
        var y:Scalar[dtype] = rho * sin(phi)
        return Vector3D(x,y,z)

    @staticmethod
    fn fromlist(iterable: List[Scalar[dtype]]) -> Self:
        if len(iterable) == 3:
            return Vector3D(iterable[0], iterable[1], iterable[2])
        else:
            var err:Error = Error()
            print(err)

    # TODO : Implement @property decorator for x,y,z once available in Mojo
    fn x(inout self, x:Scalar[dtype]):
        self._ptr[0] = x

    fn x(inout self) -> Scalar[dtype]:
        return self._ptr[0]

    fn y(inout self, y:Scalar[dtype]):
        self._ptr[1] = y

    fn y(inout self) -> Scalar[dtype]:
        return self._ptr[1]

    fn z(inout self, z:Scalar[dtype]):
        self._ptr[2] = z

    fn z(inout self) -> Scalar[dtype]:
        return self._ptr[2]

    # TODO: Implement @property decorator
    fn rho(inout self) -> Scalar[dtype]:
        return sqrt(self.x()**2 + self.y()**2)

    fn mag(inout self) -> Scalar[dtype]:
        return sqrt(self._ptr[0]**2 + self._ptr[1]**2 + self._ptr[2]**2)

    fn r(inout self) -> Scalar[dtype]:
        return self.mag()

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




