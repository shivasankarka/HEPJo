from builtin.dtype import DType
from builtin.type_aliases import Origin
from memory import UnsafePointer
from memory import memset_zero, memcpy
from sys import simdwidthof

from collections.vector import InlinedFixedVector
from algorithm import vectorize

from math import sqrt, acos, atan2, sinh, log, sin, cos, tan
import . math_funcs as mf

# Modules
from .vector3d import Vector3D
from ..constants import pi


# add support for arbitrary metric
struct LorentzVector[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
    # Fields
    var _buf: UnsafePointer[Scalar[dtype]]
    """4D vector data."""
    alias size: Int = 4
    """The size of the Vector."""
    # alias metric: StaticIntTuple[4]  = StaticIntTuple[4](-1, 1, 1, 1) if sign == -1 else StaticIntTuple[4](1, -1, -1, -1)

    """ LIFETIME METHODS """

    @always_inline("nodebug")
    fn __init__(mut self):
        """
        Initializes a Lorentz vector with zero elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)

    @always_inline("nodebug")
    fn __init__(mut self, *data: Scalar[dtype]) raises:
        """
        Initializes a Lorentz vector with the given elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 4")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size):
            self._buf[i] = data[i]

    @always_inline("nodebug")
    fn __init__(mut self, data: List[Scalar[dtype]]) raises:
        """
        Initializes a Lorentz vector with the given List of elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 4")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size):
            self._buf[i] = data[i]

    fn __init__(
        mut self,
        x: Scalar[dtype],
        y: Scalar[dtype],
        z: Scalar[dtype],
        t: Scalar[dtype],
    ):
        """
        Initializes a Lorentz vector with the given elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self._buf[0] = x
        self._buf[1] = y
        self._buf[2] = z
        self._buf[3] = t

    fn __init__(mut self, vector: Vector3D[dtype], t: Scalar[dtype]) raises:
        """
        Initializes a Lorentz vector from a 3D vector and t component.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self._buf[0] = vector[0]
        self._buf[1] = vector[1]
        self._buf[2] = vector[2]
        self._buf[3] = t

    fn __init__(mut self, vector: LorentzVector[dtype]) raises:
        """
        Initializes a Lorentz vector from another LorentzVector.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(self._buf, vector._buf, self.size)

    fn __copyinit__(out self, other: LorentzVector[dtype]):
        """
        Initializes a Lorentz vector as a copy of another vector.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(self._buf, other._buf, self.size)

    fn __moveinit__(mut self, owned other: LorentzVector[dtype]):
        """
        Initializes a LorentzVector vector by moving the data from another vector.
        """
        self._buf = other._buf

    fn __del__(owned self):
        self._buf.free()

    """ GETTER & SETTER METHODS """

    fn __getitem__(self, index: Int) raises -> Scalar[dtype]:
        if index >= 4:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            return self._buf.load(index + self.size)
        else:
            return self._buf.load(index)

    fn __setitem__(mut self, index: Int, value: Scalar[dtype]) raises:
        if index >= 4:
            raise Error("Invalid index: index exceeds size")
        self._buf.store(index, value)

    ### TRAITS ###
    fn __str__(self) -> String:
        """
        To print the LorentzVector.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            var printStr: String = "LorentzVector: ["
            for i in range(self.size):
                printStr += str(self[i])
                if i != 3:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + str(dtype)
            writer.write(printStr)
        except e:
            writer.write("Cannot convert array to string")

    fn print(self) raises -> None:
        """Prints the LorentzVector."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(self) -> String:
        """Compute the "official" string representation of LorentzVector."""
        return (
            "LorentzVector[DType."
            + str(dtype)
            + "](x="
            + str(self._buf[0])
            + ", y="
            + str(self._buf[1])
            + ", z="
            + str(self._buf[2])
            + ", t="
            + str(self._buf[3])
            + ")"
        )

    fn __len__(self) -> Int:
        """Returns the length of the LorentzVector (=4)."""
        return self.size

    fn __iter__(self) raises -> _lorentzvectorIter[__origin_of(self), dtype]:
        """Iterate over elements of the LorentzVector, returning copied value.

        Returns:
            An iterator of LorentzVector elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _lorentzvectorIter[__origin_of(self), dtype](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _lorentzvectorIter[__origin_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the LorentzVector, returning
        copied value.

        Returns:
            A reversed iterator of LorentzVector elements.
        """

        return _lorentzvectorIter[__origin_of(self), dtype, forward=False](
            array=self,
            length=self.size,
        )

    fn load[width: Int = 1](self, idx: Int) -> SIMD[dtype, width]:
        """
        SIMD load elements.
        """
        return self._buf.load[width=width](idx)

    fn store[width: Int = 1](mut self, idx: Int, val: SIMD[dtype, width]):
        """
        SIMD store elements.
        """
        self._buf.store(idx, val)

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        """
        Retreive pointer without taking ownership.
        """
        return self._buf

    fn typeof(mut self) -> DType:
        return dtype

    fn typeof_str(mut self) -> String:
        return dtype.__str__()

    """COMPARISIONS."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise equivalence.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__eq__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__eq__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(
        self, other: LorentzVector[dtype]
    ) raises -> LorentzVector[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ne__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ne__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(
        self, other: LorentzVector[dtype]
    ) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__lt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__lt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(
        self, other: LorentzVector[dtype]
    ) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__le__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__le__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(
        self, other: LorentzVector[dtype]
    ) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__gt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__gt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(
        self, other: LorentzVector[dtype]
    ) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ge__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        var result: LorentzVector[DType.bool] = LorentzVector[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ge__](
            self._buf, other, result._buf
        )
        return result

    """ARITHMETIC."""

    fn __pos__(self) raises -> Self:
        """
        Unary positve returens self unless boolean type.
        """
        return self * Scalar[dtype](1)

    fn __neg__(self) raises -> Self:
        """
        Unary negative returens self unless boolean type.
        """
        return self * Scalar[dtype](-1)

    fn __add__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__add__](
            self._buf, other, result._buf
        )
        return result^

    fn __add__(self, other: Self) -> Self:
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__add__](
            self._buf, other._buf, result._buf
        )
        return result^

    fn __radd__(mut self, other: Scalar[dtype]) -> Self:
        return self + other

    fn __radd__(self, other: Self) -> Self:
        return self + other

    fn __iadd__(mut self, other: Scalar[dtype]):
        self = self + other

    fn __iadd__(mut self, other: Self):
        self = self + other

    fn __sub__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__sub__](
            self._buf, other, result._buf
        )
        return result^

    fn __sub__(self, other: Self) -> Self:
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__sub__](
            self._buf, other._buf, result._buf
        )
        return result^

    fn __rsub__(self, other: Scalar[dtype]) raises -> Self:
        return -(self - other)

    fn __rsub__(self, other: Self) raises -> Self:
        return -(self - other)

    fn __isub__(mut self, other: Scalar[dtype]):
        self = self - other

    fn __isub__(mut self, other: Self):
        self = self - other

    fn __mul__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__mul__](
            self._buf, other, result._buf
        )
        return result^

    fn __mul__(self, other: Self) -> Self:
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__mul__](
            self._buf, other._buf, result._buf
        )
        return result^

    fn __rmul__(self, other: Scalar[dtype]) -> Self:
        return self * other

    fn __rmul__(self, other: Self) -> Self:
        return self * other

    fn __imul__(mut self, other: Scalar[dtype]):
        self = self * other

    fn __imul__(mut self, other: Self):
        self = self * other

    fn __pow__(self, p: Int) -> Self:
        return self._elementwise_pow(p)

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn _elementwise_pow(self, p: Int) -> Self:
        alias simd_width: Int = simdwidthof[dtype]()
        var new_vec = Self()

        @parameter
        fn tensor_scalar_vectorize[simd_width: Int](idx: Int) -> None:
            new_vec._buf.store(idx, pow(self._buf.load(idx), p))

        vectorize[tensor_scalar_vectorize, simd_width](self.size)
        return new_vec

    fn __truediv__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__truediv__](
            self._buf, other, result._buf
        )
        return result^

    fn __truediv__(self, other: Self) -> Self:
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__truediv__](
            self._buf, other._buf, result._buf
        )
        return result^

    fn __rtruediv__(self, other: Scalar[dtype]) -> Self:
        return self.__truediv__(other)

    fn __rtruediv__(self, other: Self) -> Self:
        return self.__truediv__(other)

    fn __itruediv__(mut self, other: Scalar[dtype]):
        self = self.__truediv__(other)

    fn __itruediv__(mut self, other: Self):
        self = self.__truediv__(other)

    # * since "*" already does element wise calculation, It's redundant for 1D array, but I could use it for dot products
    fn __matmul__(mut self, other: Self) -> Scalar[dtype]:
        return (
            self._buf[3] * other._buf[3]
            - (
                self._buf.load[width=2](0) * other._buf.load[width=2](0)
            ).reduce_add()
            - self._buf.load(2) * other._buf.load(2)
        )

    # * STATIC METHODS
    @staticmethod
    fn origin[dtype: DType = DType.float64]() -> LorentzVector[dtype]:
        return LorentzVector[dtype](0.0, 0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint[
        dtype: DType = DType.float64
    ](
        x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype], t: Scalar[dtype]
    ) -> LorentzVector[dtype]:
        return LorentzVector[dtype](x=x, y=y, z=z, t=t)

    @staticmethod
    fn fromvector[
        dtype: DType = DType.float64
    ](vector: LorentzVector[dtype]) raises -> LorentzVector[dtype]:
        return LorentzVector[dtype](vector)

    @staticmethod
    fn fromsphericalcoords[
        dtype: DType = DType.float64
    ](
        r: Scalar[dtype],
        theta: Scalar[dtype],
        phi: Scalar[dtype],
        t: Scalar[dtype],
    ) -> LorentzVector[dtype]:
        var x: Scalar[dtype] = r * sin(theta) * cos(phi)
        var y: Scalar[dtype] = r * sin(theta) * sin(phi)
        var z: Scalar[dtype] = r * cos(theta)
        return LorentzVector[dtype](x, y, z, t)

    @staticmethod
    fn fromcylindricalcoodinates[
        dtype: DType = DType.float64
    ](
        rho: Scalar[dtype],
        phi: Scalar[dtype],
        z: Scalar[dtype],
        t: Scalar[dtype],
    ) -> LorentzVector[dtype]:
        var x: Scalar[dtype] = rho * cos(phi)
        var y: Scalar[dtype] = rho * sin(phi)
        return LorentzVector[dtype](x, y, z, t)

    @staticmethod
    fn fromlist[
        dtype: DType = DType.float64
    ](iterable: List[Scalar[dtype]]) raises -> LorentzVector[dtype]:
        if len(iterable) == 4:
            return LorentzVector[dtype](
                iterable[0], iterable[1], iterable[2], iterable[3]
            )
        else:
            raise Error("Iterable size does not fit a LorentzVector")

    """ PROPERTIES """

    fn x(mut self, x: Scalar[dtype]):
        """
        Sets the x-component of the vector.

        Args:
            x: The new value for the x-component.
        """
        self._buf[0] = x

    fn x(self) -> Scalar[dtype]:
        """
        Returns the x-component of the vector.

        Returns:
            The value of the x-component.
        """
        return self._buf[0]

    fn y(mut self, y: Scalar[dtype]):
        """
        Sets the y-component of the vector.

        Args:
            y: The new value for the y-component.
        """
        self._buf[1] = y

    fn y(self) -> Scalar[dtype]:
        """
        Returns the y-component of the vector.

        Returns:
            The value of the y-component.
        """
        return self._buf[1]

    fn z(mut self, z: Scalar[dtype]):
        """
        Sets the z-component of the vector.

        Args:
            z: The new value for the z-component.
        """
        self._buf[2] = z

    fn z(self) -> Scalar[dtype]:
        """
        Returns the z-component of the vector.

        Returns:
            The value of the z-component.
        """
        return self._buf[2]

    fn t(self, t: Scalar[dtype]):
        """
        Sets the time component of the vector.
        """
        self._buf[3] = t

    fn t(self) -> Scalar[dtype]:
        """
        Returns the time component of the vector.

        Returns:
            The value of the z-component.
        """
        return self._buf[3]

    fn set(
        mut self,
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
        mut self,
        px: Scalar[dtype],
        py: Scalar[dtype],
        pz: Scalar[dtype],
        m: Scalar[dtype],
    ):
        self._buf[0] = px
        self._buf[1] = py
        self._buf[2] = pz

        if m > 0.0:
            self._buf[3] = sqrt(px**2 + py**2 + pz**2 + m**2)
        else:
            self._buf[3] = sqrt(px**2 + py**2 + pz**2 - m**2)

    fn setpxpypze(
        mut self,
        px: Scalar[dtype],
        py: Scalar[dtype],
        pz: Scalar[dtype],
        e: Scalar[dtype],
    ):
        self.set(px, py, pz, e)

    fn setptetaphim(
        mut self,
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
        mut self,
        pt: Scalar[dtype],
        eta: Scalar[dtype],
        phi: Scalar[dtype],
        e: Scalar[dtype],
    ):
        var px = pt * cos(phi)
        var py = pt * sin(phi)
        var pz = pt * sinh(eta)
        self.setpxpypze(px, py, pz, e)

    fn tolist(mut self) -> List[Scalar[dtype]]:
        return List[Scalar[dtype]](
            self._buf[0], self._buf[1], self._buf[2], self._buf[3]
        )

    fn vector(self) -> Vector3D[dtype]:
        return Vector3D[dtype](x=self._buf[0], y=self._buf[1], z=self._buf[2])

    fn mag(self) -> Scalar[dtype]:
        return sqrt(
            self._buf[3]**2 - (self._buf.load[width=2](0) ** 2).reduce_add() - self._buf.load(2)**2
        )

    fn mag2(self) -> Scalar[dtype]:
        return sqrt(
            self._buf[3]**2 - (self._buf.load[width=2](0) ** 2).reduce_add() - self._buf.load(2)**2
        )

    fn costheta(mut self) -> Scalar[dtype]:
        if self.mag() == 0.0:
            return 1.0
        else:
            return self._buf[2] / self.mag()

    fn theta(mut self, degree: Bool = False) -> Scalar[dtype]:
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180.0 / pi.cast[dtype]()
        else:
            return theta

    fn phi(mut self, degree: Bool = False) -> Scalar[dtype]:
        var phi = atan2(self._buf[1], self._buf[0])
        if degree == True:
            return phi * 180.0 / pi.cast[dtype]()
        else:
            return phi

    fn px(self) -> Scalar[dtype]:
        return self._buf[0]

    fn px(mut self, px: Scalar[dtype]):
        self._buf[0] = px

    fn py(self) -> Scalar[dtype]:
        return self._buf[1]

    fn py(mut self, py: Scalar[dtype]):
        self._buf[1] = py

    fn pz(self) -> Scalar[dtype]:
        return self._buf[2]

    fn pz(mut self, pz: Scalar[dtype]):
        self._buf[2] = pz

    fn e(self) -> Scalar[dtype]:
        return self._buf[3]

    fn e(mut self, e: Scalar[dtype]):
        self._buf[3] = e

    fn m(self) -> Scalar[dtype]:
        return self.mag()

    fn m2(self) -> Scalar[dtype]:
        return self.mag2()

    fn mass(self) -> Scalar[dtype]:
        return self.mag()

    fn mass2(self) -> Scalar[dtype]:
        return self.mag2()

    fn p(mut self) -> Scalar[dtype]:
        return self.mag()

    fn perp(mut self) -> Scalar[dtype]:
        return sqrt(self._buf[0] ** 2 + self._buf[1] ** 2)

    fn pt(mut self) -> Scalar[dtype]:
        return self.perp()

    fn et(mut self) -> Scalar[dtype]:
        return self.e() * (self.pt() / self.p())

    fn mt(mut self) -> Scalar[dtype]:
        return sqrt(self.mt2())

    fn mt2(mut self) -> Scalar[dtype]:
        return self.e() ** 2 - self.pz() ** 2

    fn beta(mut self) -> Scalar[dtype]:
        return self.p() / self.e()

    fn gamma(mut self) -> Scalar[dtype]:
        if self.beta() < 1:
            return 1.0 / sqrt(1.0 - self.beta() ** 2)
        else:
            print("Gamma > 1.0, Returning 10e10")
            return 10e10

    fn eta(mut self) -> Scalar[dtype]:
        if abs(self.costheta()) < 1.0:
            return -0.5 * log((1.0 - self.costheta()) / (1.0 + self.costheta()))
        else:
            print("eta > 1.0, Returning 10e10")
            return Scalar[dtype](10e10) if self.z() > 0 else -Scalar[dtype](
                10e10
            )

    fn pseudorapidity(mut self) -> Scalar[dtype]:
        return self.eta()

    fn rapidity(mut self) -> Scalar[dtype]:
        return 0.5 * log((self.e() + self.pz()) / (self.e() - self.pz()))

    fn copy(mut self) raises -> Self:
        return Self(self._buf[0], self._buf[1], self._buf[2], self._buf[3])

    fn boostvector(self) raises -> Vector3D[dtype]:
        return Vector3D(
            self._buf[0] / self._buf[3],
            self._buf[1] / self._buf[3],
            self._buf[2] / self._buf[3],
        )

    fn boost(self, mut args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            raise Error(
                "Boost vector must be an instance of Vector3D of size 3."
            )

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

    fn boostplus(self, mut args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            raise Error(
                "Boost vector must be an instance of Vector3D of size 3."
            )

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

    fn boostminus(self, mut args: Vector3D[dtype]) raises -> Self:
        if len(args) != 3:
            raise Error(
                "Boost vector must be an instance of Vector3D of size 3."
            )

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
    fn dot(mut self, other: Self) raises -> Scalar[dtype]:
        return self @ other

    fn isspacelike(mut self) raises -> Bool:
        if self.mag() != 0.0:
            return self.mag2() < 0.0
        else:
            raise Error("Magnitude is zero")

    fn istimelike(mut self) raises -> Bool:
        if self.mag() != 0.0:
            return self.mag2() > 0.0
        else:
            raise Error("Magnitude is zero")

    fn islightlike(mut self) -> Bool:
        return self.mag2() == 0.0

    fn torestframe(self) raises -> Self:
        var boost_vec: Vector3D[dtype] = self.boostvector()
        return self.boostplus(boost_vec)


@value
struct _lorentzvectorIter[
    is_mutable: Bool, //,
    lifetime: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for LorentzVector.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: LorentzVector[dtype]
    var length: Int

    fn __init__(
        mut self,
        array: LorentzVector[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Scalar[dtype]:
        @parameter
        if forward:
            var current_index = self.index
            self.index += 1
            return self.array.__getitem__(current_index)
        else:
            var current_index = self.index
            self.index -= 1
            return self.array.__getitem__(current_index)

    @always_inline
    fn __has_next__(self) -> Bool:
        @parameter
        if forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return self.length - self.index
        else:
            return self.index
