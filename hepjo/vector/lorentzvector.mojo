# Defaults
from builtin.dtype import DType
from collections.vector import InlinedFixedVector
from algorithm import vectorize
from math import sqrt, acos, atan2, sinh, log, sin, cos, tan
import .math_funcs as mf

# Modules
from .vector3d import Vector3D
from .constants import pi

@value
struct _lorentzvectorIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
    dtype: DType,
    sign: Int, 
    forward: Bool = True,
]:
    """Iterator for LorentzVector.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        sign: The metric signature.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: LorentzVector[dtype, sign]
    var length: Int

    fn __init__(
        inout self,
        array: LorentzVector[dtype, sign],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) raises -> Scalar[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index

@value
struct LorentzVector[dtype: DType = DType.float64, sign: Int = -1](
    Intable, CollectionElement, Sized
):

    # Fields
    var data: DTypePointer[dtype]
    """3D vector data."""
    alias size: Int = 4
    """The size of the Vector."""
    alias metric: StaticIntTuple[4]  = StaticIntTuple[4](-1, 1, 1, 1) if sign == -1 else StaticIntTuple[4](1, -1, -1, -1)

    """Default constructor."""

    fn __init__(inout self):
        """
        Initializes a 4D vector with zero elements.
        """
        self.data = DTypePointer[dtype].alloc(self.size)
        memset_zero(self.data, self.size)

    fn __init__(inout self, *data: Scalar[dtype]) raises:
        if len(data) != 4:
            raise Error("Length of given data is more than 4.")
        self.data = DTypePointer[dtype].alloc(self.size)
        memset_zero(self.data, self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    fn __init__(inout self, *data: List[Scalar[dtype]]):
        self.data = DTypePointer[dtype].alloc(self.size)
        memset_zero(self.data, self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    fn __init__(
        inout self, vector3d: Vector3D[dtype], t: Scalar[dtype]
    ) raises:
        self.data = DTypePointer[dtype].alloc(self.size)
        self.data[0] = vector3d[0]
        self.data[1] = vector3d[1]
        self.data[2] = vector3d[2]
        self.data[3] = t

    fn __init__(
        inout self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype], t: Scalar[dtype]
    ) raises:
        self.data = DTypePointer[dtype].alloc(self.size)
        self.data[0] = x
        self.data[1] = y
        self.data[2] = z
        self.data[3] = t

    fn __str__(self) -> String:
        """
        To use Stringable trait and print the array.
        """
        var printStr: String = "LorentzVector: ["
        for i in range(self.size):
            try:
                printStr += str(self[i])
            except:
                print("Cannot convery LorentzVector to string.")
            if i != 3:
                printStr += " , "

        printStr += "]" + "\n"
        printStr += "dtype=" + str(dtype) + "; " + "Sign: "
        if sign == -1:
            printStr += "-+++"
        else:
            printStr += "+---"
        return printStr
    
    fn print(self) raises -> None:
        """Prints the LorentzVector."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(inout self) -> String:
        """Compute the "official" string representation of LorentzVector."""
        return (
            "LorentzVector[DType."
            + str(dtype)
            + "](x="
            + str(self.data[0])
            + ", y="
            + str(self.data[1])
            + ", z="
            + str(self.data[2])
            + ", t="
            + str(self.data[3])
            + ")"
        )

    fn __getitem__(self, index: Int) raises -> Scalar[dtype]:
        if index > 3:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            return self.data.load[width=1](index + self.size)
        else:
            return self.data.load[width=1](index)

    fn __setitem__(inout self, index: Int, value: Scalar[dtype]) raises:
        if index > 3:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            self.data.store[width=1](index + self.size, value)
        else:
            self.data.store[width=1](index, value)

    fn __len__(self) -> Int:
        """Returns the length of the Vector3D (=3)."""
        return self.size

    # TODO: change this to represent the Int version
    fn __int__(self) -> Int:
        return self.size

    fn __iter__(self) raises -> _lorentzvectorIter[__lifetime_of(self), dtype, sign]:
        """Iterate over elements of the Vector3D, returning copied value.

        Returns:
            An iterator of Vector3D elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _lorentzvectorIter[__lifetime_of(self), dtype, sign](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _lorentzvectorIter[__lifetime_of(self), dtype, sign, forward=False]:
        """Iterate backwards over elements of the Vector3D, returning
        copied value.

        Returns:
            A reversed iterator of Vector3D elements.
        """

        return _lorentzvectorIter[__lifetime_of(self), dtype, sign, forward=False](
            array=self,
            length=self.size,
        )

    fn __del__(owned self):
        self.data.free()

    fn __pos__(self) -> Self:
        return self * (1.0)

    fn __neg__(self) -> Self:
        return self * (-1.0)

    fn load[width: Int = 1](self, idx: Int) -> SIMD[dtype, width]:
        """
        SIMD load elements.
        """
        return self.data.load[width=width](idx)

    fn store[width: Int = 1](inout self, idx: Int, val: SIMD[dtype, width]):
        """
        SIMD store elements.
        """
        self.data.store[width=width](idx, val)

    fn unsafe_ptr(self) -> DTypePointer[dtype, 0]:
        """
        Retreive pointer without taking ownership.
        """
        return self.data

    fn typeof(inout self) -> DType:
        return dtype

    fn typeof_str(inout self) -> String:
        return dtype.__str__()

    """COMPARISIONS."""
    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Vector3D[DType.bool]:
        """
        Itemwise equivelence.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__eq__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise equivelence between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__eq__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ne__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ne__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__lt__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__lt__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__le__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__le__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__gt__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__gt__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ge__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        var result:Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ge__](
            self.data, other, result.data
        )
        return result

    #
    """ARITHMETIC."""
    fn __add__(self, other: Scalar[dtype]) -> Self:
        var result:Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__add__](
            self.data, other, result.data
        )
        return result

    fn __add__(self, other: Self) -> Self:
        var result:Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__add__](
            self.data, other.data, result.data
        )
        return result

    fn __radd__(self, other: Scalar[dtype]) -> Self:
        return self + other

    fn __iadd__(inout self, other: Scalar[dtype]):
        self = self + other

    fn __sub__(self, other: Scalar[dtype]) -> Self:
        var result:Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__sub__](
            self.data, other, result.data
        )
        return result

    fn __sub__(self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__sub__](
            # self.data, other.data
        # )
        var result:Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__sub__](
            self.data, other.data, result.data
        )
        return result

    fn __rsub__(self, other: Scalar[dtype]) -> Self:
        return -(self - other)

    fn __isub__(inout self, other: Scalar[dtype]):
        self = self - other

    fn __mul__(self, other: Scalar[dtype]) -> Self:
        var result:Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__mul__](
            self.data, other, result.data
        )
        return result

    fn __mul__(self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__mul__](
        #     self.data, other.data
        # )
        var result:Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__mul__](
            self.data, other.data, result.data
        )
        return result

    fn __rmul__(self, other: Scalar[dtype]) -> Self:
        return self * other

    fn __imul__(inout self, other: Scalar[dtype]):
        self = self * other

    # * since "*" already does element wise calculation, I think matmul is redundant for 1D array, but I could use it for dot products
    # fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
    #     return self._elementwise_array_arithmetic[SIMD.__mul__](other)._reduce_sum()

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __ipow__(inout self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = Self()

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec.data.store[width=simd_width](
                idx, pow(self.data.load[width=simd_width](idx), p)
            )

        vectorize[tensor_scalar_vectorize, simd_width](self.size)
        return new_vec

    fn __truediv__(inout self, other: Scalar[dtype]) -> Self:
        var result:Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__truediv__](
            self.data, other, result.data
        )
        return result

    fn __truediv__(inout self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[
        #     self.size, dtype, SIMD.__truediv__
        # ](self.data, other.data)
        var result:Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__truediv__](
            self.data, other.data, result.data
        )
        return result

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other: Self):
        self = self.__truediv__(other)

    fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)

    # PROPERTIES
    fn x(self) -> Scalar[dtype]:
        return self.data[0]

    fn y(self) -> Scalar[dtype]:
        return self.data[1]

    fn z(self) -> Scalar[dtype]:
        return self.data[2]

    fn t(self) -> Scalar[dtype]:
        return self.data[3]

    fn vector(self) -> Vector3D[dtype]:
        return Vector3D[dtype](x= self.data[0], y= self.data[1], z= self.data[2])

    fn x(self, value: Scalar[dtype]):
        self.data[0] = value

    fn y(self, value: Scalar[dtype]):
        self.data[1] = value

    fn z(self, value: Scalar[dtype]):
        self.data[2] = value

    fn t(self, value: Scalar[dtype]):
        self.data[3] = value

    fn mag(inout self) -> Scalar[dtype]:
        return sqrt(self.x() ** 2 + self.y() ** 2 + self.z() ** 2)

    fn mag2(inout self) -> Scalar[dtype]:
        return self.x() ** 2 + self.y() ** 2 + self.z() ** 2

    fn magl2(inout self) -> Scalar[dtype]:
        return self.data[3] ** 2 - self.mag2()

    fn magl(inout self) -> Scalar[dtype]:
        var magn2 = self.magl2()
        return sqrt(magn2) if magn2 > 0.0 else -sqrt(-magn2)

    fn costheta(inout self) -> Scalar[dtype]:
        if self.mag() == 0.0:
            return 1.0
        else:
            return self.data[2] / self.mag()

    fn theta(inout self, degree: Bool = False) -> Scalar[dtype]:
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / Scalar[dtype](pi)
        else:
            return theta

    fn phi(inout self, degree: Bool = False) -> Scalar[dtype]:
        var phi = atan2(self.data[1], self.data[0])
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

    fn m(inout self) -> Scalar[dtype]:
        return sqrt(self.e() ** 2 - self.mag2())

    fn set(
        inout self,
        x: Scalar[dtype],
        y: Scalar[dtype],
        z: Scalar[dtype],
        t: Scalar[dtype],
    ):
        self.x(x)
        self.y(y)
        self.z(z)
        self.t(t)

    fn setpxpypzm(
        inout self,
        px: Scalar[dtype],
        py: Scalar[dtype],
        pz: Scalar[dtype],
        m: Scalar[dtype],
    ):
        self.data[0] = px
        self.data[1] = py
        self.data[2] = pz

        if m > 0.0:
            self.data[3] = sqrt(px**2 + py**2 + pz**2 + m**2)
        else:
            self.data[3] = sqrt(px**2 + py**2 + pz**2 - m**2)

    fn setpxpypze(
        inout self,
        px: Scalar[dtype],
        py: Scalar[dtype],
        pz: Scalar[dtype],
        e: Scalar[dtype],
    ):
        self.set(px, py, pz, e)

    fn setptetaphim(
        inout self,
        pt: Scalar[dtype],
        eta: Scalar[dtype],
        phi: Scalar[dtype],
        m: Scalar[dtype],
    ):
        var px = pt * cos(phi)
        var py = pt * sin(phi)
        var pz = pt * sinh(eta)
        self.setpxpypzm(px, py, pz, m)

    fn setptetaphie(
        inout self,
        pt: Scalar[dtype],
        eta: Scalar[dtype],
        phi: Scalar[dtype],
        e: Scalar[dtype],
    ):
        var px = pt * cos(phi)
        var py = pt * sin(phi)
        var pz = pt * sinh(eta)
        self.setpxpypze(px, py, pz, e)

    fn tolist(inout self) -> List[Scalar[dtype]]:
        return List[Scalar[dtype]](
            self.data[0], self.data[1], self.data[2], self.data[3]
        )

    fn p(inout self) -> Scalar[dtype]:
        return self.mag()

    fn perp(inout self) -> Scalar[dtype]:
        return sqrt(self.data[0] ** 2 + self.data[1] ** 2)

    fn pt(inout self) -> Scalar[dtype]:
        return self.perp()

    fn et(inout self) -> Scalar[dtype]:
        return self.e() * (self.pt() / self.p())

    fn minv(inout self) -> Scalar[dtype]:
        return self.magl()

    fn mt(inout self) -> Scalar[dtype]:
        return sqrt(self.mt2())

    fn mt2(inout self) -> Scalar[dtype]:
        return self.e() ** 2 - self.pz() ** 2

    fn beta(inout self) -> Scalar[dtype]:
        return self.p() / self.e()

    fn gamma(inout self) -> Scalar[dtype]:
        if self.beta() < 1:
            return 1.0 / sqrt(1.0 - self.beta() ** 2)
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

    fn copy(inout self) raises -> Self:
        return Self(self.data[0], self.data[1], self.data[2], self.data[3])

    # Implement iter
    fn boostvector(inout self) -> Vector3D[dtype]:
        return Vector3D(
            self.x() / self.t(), self.y() / self.t(), self.z() / self.t()
        )

    fn boost(inout self, inout args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx: Scalar[dtype] = args[0]
        var by: Scalar[dtype] = args[1]
        var bz: Scalar[dtype] = args[2]

        var b2: Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma: Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp: Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2: Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp: Scalar[
            dtype
        ] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp: Scalar[
            dtype
        ] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp: Scalar[
            dtype
        ] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    fn boostplus(inout self, inout args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx: Scalar[dtype] = args[0]
        var by: Scalar[dtype] = args[1]
        var bz: Scalar[dtype] = args[2]

        var b2: Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma: Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp: Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2: Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp: Scalar[
            dtype
        ] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp: Scalar[
            dtype
        ] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp: Scalar[
            dtype
        ] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    fn boostminus(inout self, inout args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            print("Error, it is not a valid vector size")

        var bx: Scalar[dtype] = -1.0 * args[0]
        var by: Scalar[dtype] = -1.0 * args[1]
        var bz: Scalar[dtype] = -1.0 * args[2]

        var b2: Scalar[dtype] = bx**2 + by**2 + bz**2
        var gamma: Scalar[dtype] = 1.0 / sqrt(1.0 - b2)
        var bp: Scalar[dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2: Scalar[dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp: Scalar[
            dtype
        ] = self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        var yp: Scalar[
            dtype
        ] = self.y() + gamma2 * bp * by - gamma * by * self.t()
        var zp: Scalar[
            dtype
        ] = self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    # maybe you can change this implementation
    fn dot(inout self, other: Self) raises -> Scalar[dtype]:
        if sign == -1:
            var metric = LorentzVector[dtype, sign](-1, 1, 1, 1)
            return (self * (other * metric))._reduce_sum()
            # return - self.t() * other.t() + self.vector().dot(other.vector())
        else:
            var metric = LorentzVector[dtype, sign](1, -1, -1, -1)
            return (self * (other * metric))._reduce_sum()
            # return self.t() * other.t() - self.vector().dot(other.vector())

    fn _reduce_sum(self) -> Scalar[dtype]:    
       """
        Computes the sum of all elements in the vector using SIMD operations for efficiency.

        This function performs a reduction operation to sum all elements of the vector. It leverages SIMD capabilities to load and add multiple elements simultaneously, which can significantly speed up the operation on large vectors. The result is a scalar value representing the sum of all elements.

        Returns:
            A scalar of type `dtype` representing the sum of all elements in the vector.
        """
        var reduced: Scalar[dtype] = Scalar[dtype](0)
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced += self.data.load[width=simd_width](idx).reduce_add()

        vectorize[vectorize_reduce, simd_width](self.size)
        return reduced


    fn isspacelike(inout self) raises -> Bool:
        if self.magl2() != 0.0:
            if sign == -1:
                return self.magl2() > 0.0
            else:
                return self.magl2() < 0.0
        else:
            raise Error("Magnitude is zero")

    fn istimelike(inout self) raises -> Bool:
        if self.magl2() != 0.0:
            if sign == -1:
                return self.magl2() < 0.0
            else:
                return self.magl2() > 0.0
        else:
            raise Error("Magnitude is zero")

    fn islightlike(inout self) -> Bool:
        return self.magl2() == 0.0

    fn torestframe(inout self) raises -> Self:
        var boost_vec = self.boostvector()
        return self.boostplus(boost_vec)
