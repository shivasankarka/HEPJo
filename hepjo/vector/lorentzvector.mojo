from math import sqrt, acos, atan2, sinh, log, sin, cos, tan

from .vector3d import Vector3D
from ..constants import pi

################################################################################################################
####################################### LORENTZ VECTOR ##########################################################
################################################################################################################

# ===----------------------------------------------------------------------===#
# FORMAT FOR DOCSTRING (See "Mojo docstring style guide" for more information)
# 1. Description *
# 2. Parameters *
# 3. Args *
# 4. Constraints *
# 4) Returns *
# 5) Raises *
# 6) SEE ALSO
# 7) NOTES
# 8) REFERENCES
# 9) Examples *
# (Items marked with * are flavored in "Mojo docstring style guide")
# ===----------------------------------------------------------------------===#

# TODO: Add support for both metric signatures (+---) and (-+++)
struct LorentzVector[dtype: DType = DType.float64](
    ImplicitlyCopyable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    comptime size: Int = 4
    """The size of the Vector."""

    # Fields
    var _t: Scalar[Self.dtype]
    var _x: Scalar[Self.dtype]
    var _y: Scalar[Self.dtype]
    var _z: Scalar[Self.dtype]
    """4D Lorentz vector data."""

    # """LIFETIME METHODS"""
    @always_inline("nodebug")
    fn __init__(out self):
        """
        Initializes a Lorentz vector with zero elements.
        """
        self._x = 0
        self._y = 0
        self._z = 0
        self._t = 0

    @always_inline("nodebug")
    fn __init__(out self, data: List[Scalar[Self.dtype]]) raises:
        """
        Initializes a Lorentz vector with the given List of elements.

        Args:
            data: Iterable with 4 elements representing (x, y, z, t).
        """
        if len(data) != self.size:
            raise Error("Length of input should be 4")
        self._x = data[0]
        self._y = data[1]
        self._z = data[2]
        self._t = data[3]

    @always_inline("nodebug")
    fn __init__(out self, *data: Scalar[Self.dtype]) raises:
        """
        Initializes a Lorentz vector with the given elements (varargs).
        """
        if len(data) != self.size:
            raise Error("Length of input should be 4")
        self._x = data[0]
        self._y = data[1]
        self._z = data[2]
        self._t = data[3]

    @always_inline("nodebug")
    fn __init__(
        out self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
        t: Scalar[Self.dtype],
    ):
        """
        Initializes a Lorentz vector with the given components.
        """
        self._x = x
        self._y = y
        self._z = z
        self._t = t

    @always_inline("nodebug")
    fn __init__(out self, vector: Vector3D[Self.dtype], t: Scalar[Self.dtype]) raises:
        """
        Initializes a Lorentz vector from a 3D vector and t component.
        """
        self._x = vector[0]
        self._y = vector[1]
        self._z = vector[2]
        self._t = t

    @always_inline("nodebug")
    fn __init__(out self, vector: LorentzVector[Self.dtype]) raises:
        """
        Initializes a Lorentz vector from another LorentzVector.
        """
        self._x = vector._x
        self._y = vector._y
        self._z = vector._z
        self._t = vector._t

    fn __copyinit__(out self, other: Self):
        """
        Initializes a Lorentz vector as a copy of another vector.
        """
        self._x = other._x
        self._y = other._y
        self._z = other._z
        self._t = other._t

    # """GETTER & SETTER METHODS"""
    fn __getitem__(self, var index: Int) raises -> Scalar[Self.dtype]:
        if index >= 4:
            raise Error("Invalid index: index exceeds size")
        if index < 0:
            index = index + self.size
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        elif index == 2:
            return self._z
        else:
            return self._t

    fn __getattr__[name: StringLiteral](self) raises -> Scalar[Self.dtype]:
        if name == "x":
            return self._x
        elif name == "y":
            return self._y
        elif name == "z":
            return self._z
        elif name == "t":
            return self._t
        elif name == "px":
            return self._x
        elif name == "py":
            return self._y
        elif name == "pz":
            return self._z
        elif name == "e":
            return self._t
        elif name == "m":
            return self.mag()
        elif name == "mass":
            return self.mag()
        elif name == "mass2":
            return self.mag2()
        else:
            raise Error(
                "AttributeError: 'LorentzVector' object has no attribute '" + name + "'"
            )

    fn __setitem__(mut self, index: Int, value: Scalar[Self.dtype]) raises:
        if index >= 4:
            raise Error("Invalid index: index exceeds size")
        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        elif index == 2:
            self._z = value
        else:
            self._t = value

    ### TRAITS ###
    fn __str__(self) -> String:
        """
        To print the LorentzVector.
        """
        return String.write(self)

    # TODO: remove string allocs by writing to writer directly.
    fn write_to[W: Writer](self, mut writer: W):
        try:
            var printStr: String = "LorentzVector: ["
            for i in range(self.size):
                printStr += String(self[i])
                if i != 3:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + String(Self.dtype)
            writer.write(printStr)
        except e:
            writer.write("Cannot convert array to string")

    fn print(self) raises -> None:
        """Prints the LorentzVector."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(self) -> String:
        """Compute the \"official\" string representation of LorentzVector."""
        return (
            "LorentzVector[DType."
            + String(Self.dtype)
            + "](x="
            + String(self._x)
            + ", y="
            + String(self._y)
            + ", z="
            + String(self._z)
            + ", t="
            + String(self._t)
            + ")"
        )

    fn __len__(self) -> Int:
        """Returns the length of the LorentzVector (=4)."""
        return self.size

    fn __iter__(self) raises -> _lorentzvectorIter[origin_of(self), Self.dtype]:
        """Iterate over elements of the LorentzVector, returning copied value.

        Returns:
            An iterator of LorentzVector elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _lorentzvectorIter[origin_of(self), Self.dtype](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _lorentzvectorIter[origin_of(self), Self.dtype, forward=False]:
        """Iterate backwards over elements of the LorentzVector, returning
        copied value.

        Returns:
            A reversed iterator of LorentzVector elements.
        """

        return _lorentzvectorIter[origin_of(self), Self.dtype, forward=False](
            array=self,
            length=self.size,
        )

    fn typeof(mut self) -> DType:
        return Self.dtype

    fn typeof_str(mut self) -> String:
        return Self.dtype.__str__()

    # """COMPARISIONS."""
    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise equivalence.
        """
        return LorentzVector[DType.bool](
            self._x == other._x,
            self._y == other._y,
            self._z == other._z,
            self._t == other._t,
        )

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise equivalence between scalar and Vector.
        """
        return LorentzVector[DType.bool](
            self._x == other,
            self._y == other,
            self._z == other,
            self._t == other,
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise nonequivalence.
        """
        return LorentzVector[DType.bool](
            self._x != other._x,
            self._y != other._y,
            self._z != other._z,
            self._t != other._t,
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise nonequivalence.
        """
        return LorentzVector[DType.bool](
            self._x != other,
            self._y != other,
            self._z != other,
            self._t != other,
        )

    @always_inline("nodebug")
    fn __lt__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than between vectors.
        """
        return LorentzVector[DType.bool](
            self._x < other._x,
            self._y < other._y,
            self._z < other._z,
            self._t < other._t,
        )

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than between vector and scalar.
        """
        return LorentzVector[DType.bool](
            self._x < other,
            self._y < other,
            self._z < other,
            self._t < other,
        )

    @always_inline("nodebug")
    fn __le__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than or equal to between vectors.
        """
        return LorentzVector[DType.bool](
            self._x <= other._x,
            self._y <= other._y,
            self._z <= other._z,
            self._t <= other._t,
        )

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise less than or equal to between vector and scalar.
        """
        return LorentzVector[DType.bool](
            self._x <= other,
            self._y <= other,
            self._z <= other,
            self._t <= other,
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than between vectors.
        """
        return LorentzVector[DType.bool](
            self._x > other._x,
            self._y > other._y,
            self._z > other._z,
            self._t > other._t,
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than between vector and scalar.
        """
        return LorentzVector[DType.bool](
            self._x > other,
            self._y > other,
            self._z > other,
            self._t > other,
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Self) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than or equal to between vectors.
        """
        return LorentzVector[DType.bool](
            self._x >= other._x,
            self._y >= other._y,
            self._z >= other._z,
            self._t >= other._t,
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[Self.dtype]) raises -> LorentzVector[DType.bool]:
        """
        Itemwise greater than or equal to between vector and scalar.
        """
        return LorentzVector[DType.bool](
            self._x >= other,
            self._y >= other,
            self._z >= other,
            self._t >= other,
        )

    # """ARITHMETIC."""
    fn __pos__(self) raises -> Self:
        """
        Unary positive returns self.
        """
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returns -self.
        """
        return self * Scalar[Self.dtype](-1)

    fn __add__(self, other: Scalar[Self.dtype]) -> Self:
        return Self(
            self._x + other,
            self._y + other,
            self._z + other,
            self._t + other,
        )

    fn __add__(self, other: Self) -> Self:
        return Self(
            self._x + other._x,
            self._y + other._y,
            self._z + other._z,
            self._t + other._t,
        )

    fn __radd__(mut self, other: Scalar[Self.dtype]) -> Self:
        return self + other

    fn __radd__(self, other: Self) -> Self:
        return self + other

    fn __iadd__(mut self, other: Scalar[Self.dtype]):
        self = self + other

    fn __iadd__(mut self, other: Self):
        self = self + other

    fn __sub__(self, other: Scalar[Self.dtype]) -> Self:
        return Self(
            self._x - other,
            self._y - other,
            self._z - other,
            self._t - other,
        )

    fn __sub__(self, other: Self) -> Self:
        return Self(
            self._x - other._x,
            self._y - other._y,
            self._z - other._z,
            self._t - other._t,
        )

    fn __rsub__(self, other: Scalar[Self.dtype]) raises -> Self:
        return -(self - other)

    fn __rsub__(self, other: Self) raises -> Self:
        return -(self - other)

    fn __isub__(mut self, other: Scalar[Self.dtype]):
        self = self - other

    fn __isub__(mut self, other: Self):
        self = self - other

    fn __mul__(self, other: Scalar[Self.dtype]) -> Self:
        return Self(
            self._x * other,
            self._y * other,
            self._z * other,
            self._t * other,
        )

    fn __mul__(self, other: Self) -> Self:
        return Self(
            self._x * other._x,
            self._y * other._y,
            self._z * other._z,
            self._t * other._t,
        )

    fn __rmul__(self, other: Scalar[Self.dtype]) -> Self:
        return self * other

    fn __rmul__(self, other: Self) -> Self:
        return self * other

    fn __imul__(mut self, other: Scalar[Self.dtype]):
        self = self * other

    fn __imul__(mut self, other: Self):
        self = self * other

    fn __pow__(self, p: Int) -> Self:
        return Self(self._x**p, self._y**p, self._z**p, self._t**p)

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        if other == 0:
            raise Error("Division by zero error in LorentzVector.__truediv__")
        return Self(
            self._x / other,
            self._y / other,
            self._z / other,
            self._t / other,
        )

    fn __truediv__(self, other: Self) raises -> Self:
        if other._x == 0 or other._y == 0 or other._z == 0 or other._t == 0:
            raise Error("Division by zero error in LorentzVector.__truediv__")
        return Self(
            self._x / other._x,
            self._y / other._y,
            self._z / other._z,
            self._t / other._t,
        )

    fn __rtruediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.__truediv__(other)

    fn __rtruediv__(self, other: Self) raises -> Self:
        return self.__truediv__(other)

    fn __itruediv__(mut self, other: Scalar[Self.dtype]) raises:
        self = self.__truediv__(other)

    fn __itruediv__(mut self, other: Self) raises:
        self = self.__truediv__(other)

    fn __matmul__(self, other: Self) -> Scalar[Self.dtype]:
        """
        Minkowski inner product: t1*t2 - x1*x2 - y1*y2 - z1*z2.
        """
        return (
            self._t * other._t
            - (self._x * other._x + self._y * other._y + self._z * other._z)
        )

    # * STATIC METHODS
    @staticmethod
    fn origin() -> Self:
        return Self(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(
        x: Scalar[Self.dtype], y: Scalar[Self.dtype], z: Scalar[Self.dtype], t: Scalar[Self.dtype]
    ) -> Self:
        return Self(x=x, y=y, z=z, t=t)

    @staticmethod
    fn fromvector(v: Self) raises -> Self:
        return Self(v._x, v._y, v._z, v._t)

    @staticmethod
    fn fromsphericalcoords(
        r: Scalar[Self.dtype],
        theta: Scalar[Self.dtype],
        phi: Scalar[Self.dtype],
        t: Scalar[Self.dtype],
    ) -> Self:
        var x: Scalar[Self.dtype] = r * sin(theta) * cos(phi)
        var y: Scalar[Self.dtype] = r * sin(theta) * sin(phi)
        var z: Scalar[Self.dtype] = r * cos(theta)
        return Self(x, y, z, t)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[Self.dtype], phi: Scalar[Self.dtype], z: Scalar[Self.dtype], t: Scalar[Self.dtype]
    ) -> Self:
        var x: Scalar[Self.dtype] = rho * cos(phi)
        var y: Scalar[Self.dtype] = rho * sin(phi)
        return Self(x, y, z, t)

    @staticmethod
    fn fromlist(iterable: List[Scalar[Self.dtype]]) raises -> Self:
        if len(iterable) == 4:
            return Self(iterable[0], iterable[1], iterable[2], iterable[3])
        else:
            raise Error("Iterable size does not fit a LorentzVector")

    # * PROPERTIES
    fn x(mut self, x: Scalar[Self.dtype]):
        """
        Sets the x-component of the vector.

        Args:
            x: The new value for the x-component.
        """
        self._x = x

    fn x(self) -> Scalar[Self.dtype]:
        """
        Returns the x-component of the vector.

        Returns:
            The value of the x-component.
        """
        return self._x

    fn y(mut self, y: Scalar[Self.dtype]):
        """
        Sets the y-component of the vector.

        Args:
            y: The new value for the y-component.
        """
        self._y = y

    fn y(self) -> Scalar[Self.dtype]:
        """
        Returns the y-component of the vector.

        Returns:
            The value of the y-component.
        """
        return self._y

    fn z(mut self, z: Scalar[Self.dtype]):
        """
        Sets the z-component of the vector.

        Args:
            z: The new value for the z-component.
        """
        self._z = z

    fn z(self) -> Scalar[Self.dtype]:
        """
        Returns the z-component of the vector.

        Returns:
            The value of the z-component.
        """
        return self._z

    fn t(mut self, t: Scalar[Self.dtype]):
        """
        Sets the time/energy component of the vector.
        """
        self._t = t

    fn t(self) -> Scalar[Self.dtype]:
        """
        Returns the time/energy component of the vector.

        Returns:
            The value of the t-component.
        """
        return self._t

    fn set(
        mut self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
        t: Scalar[Self.dtype],
    ):
        """
        Sets all 4 components of the vector.
        """
        self._x = x
        self._y = y
        self._z = z
        self._t = t

    fn setpxpypzm(
        mut self,
        px: Scalar[Self.dtype],
        py: Scalar[Self.dtype],
        pz: Scalar[Self.dtype],
        m: Scalar[Self.dtype],
    ):
        """
        Set (px, py, pz, mass) and compute energy accordingly.
        """
        self._x = px
        self._y = py
        self._z = pz

        if m > 0.0:
            self._t = sqrt(px**2 + py**2 + pz**2 + m**2)
        else:
            self._t = sqrt(px**2 + py**2 + pz**2 - m**2)

    fn setpxpypze(
        mut self,
        px: Scalar[Self.dtype],
        py: Scalar[Self.dtype],
        pz: Scalar[Self.dtype],
        e: Scalar[Self.dtype],
    ):
        self.set(px, py, pz, e)

    fn setptetaphim(
        mut self,
        pt: Scalar[Self.dtype],
        eta: Scalar[Self.dtype],
        phi: Scalar[Self.dtype],
        m: Scalar[Self.dtype],
    ):
        var px = pt * cos(phi)
        var py = pt * sin(phi)
        var pz = pt * sinh(eta)
        self.setpxpypzm(px, py, pz, m)

    fn setptetaphie(
        mut self,
        pt: Scalar[Self.dtype],
        eta: Scalar[Self.dtype],
        phi: Scalar[Self.dtype],
        e: Scalar[Self.dtype],
    ):
        var px = pt * cos(phi)
        var py = pt * sin(phi)
        var pz = pt * sinh(eta)
        self.setpxpypze(px, py, pz, e)

    fn tolist(mut self) -> List[Scalar[Self.dtype]]:
        """
        Converts the vector components to a list in (x, y, z, t) order.
        """
        return [
            Scalar[Self.dtype](self._x),
            self._y,
            self._z,
            self._t,
        ]

    fn vector(self) -> Vector3D[Self.dtype]:
        """
        Returns the spatial 3-vector (px, py, pz).
        """
        return Vector3D[Self.dtype](x=self._x, y=self._y, z=self._z)

    fn mag(self) -> Scalar[Self.dtype]:
        """
        Calculates the invariant magnitude (mass) of the Lorentz vector: sqrt(t^2 - x^2 - y^2 - z^2).
        """
        return sqrt(self._t**2 - (self._x**2 + self._y**2 + self._z**2))

    fn mag2(self) -> Scalar[Self.dtype]:
        """
        Returns the squared invariant mass: t^2 - x^2 - y^2 - z^2.
        """
        return self._t**2 - (self._x**2 + self._y**2 + self._z**2)

    fn costheta(mut self) -> Scalar[Self.dtype]:
        if self.mag() == 0.0:
            return 1.0
        else:
            return self._z / self.mag()

    fn theta(mut self, degree: Bool = False) -> Scalar[Self.dtype]:
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180.0 / pi.cast[Self.dtype]()
        else:
            return theta

    fn phi(mut self, degree: Bool = False) -> Scalar[Self.dtype]:
        var phi = atan2(self._y, self._x)
        if degree == True:
            return phi * 180.0 / pi.cast[Self.dtype]()
        else:
            return phi

    fn px(self) -> Scalar[Self.dtype]:
        return self._x

    fn px(mut self, px: Scalar[Self.dtype]):
        self._x = px

    fn py(self) -> Scalar[Self.dtype]:
        return self._y

    fn py(mut self, py: Scalar[Self.dtype]):
        self._y = py

    fn pz(self) -> Scalar[Self.dtype]:
        return self._z

    fn pz(mut self, pz: Scalar[Self.dtype]):
        self._z = pz

    fn e(self) -> Scalar[Self.dtype]:
        return self._t

    fn e(mut self, e: Scalar[Self.dtype]):
        self._t = e

    fn m(self) -> Scalar[Self.dtype]:
        return self.mag()

    fn m2(self) -> Scalar[Self.dtype]:
        return self.mag2()

    fn mass(self) -> Scalar[Self.dtype]:
        return self.mag()

    fn mass2(self) -> Scalar[Self.dtype]:
        return self.mag2()

    fn p(mut self) -> Scalar[Self.dtype]:
        return sqrt(self._x**2 + self._y**2 + self._z**2)

    fn perp(mut self) -> Scalar[Self.dtype]:
        return sqrt(self._x ** 2 + self._y ** 2)

    fn pt(mut self) -> Scalar[Self.dtype]:
        return self.perp()

    fn et(mut self) -> Scalar[Self.dtype]:
        return self.e() * (self.pt() / self.p())

    fn mt(mut self) -> Scalar[Self.dtype]:
        return sqrt(self.mt2())

    fn mt2(mut self) -> Scalar[Self.dtype]:
        return self.e() ** 2 - self.pz() ** 2

    fn beta(mut self) -> Scalar[Self.dtype]:
        return self.p() / self.e()

    fn gamma(mut self) -> Scalar[Self.dtype]:
        if self.beta() < 1:
            return 1.0 / sqrt(1.0 - self.beta() ** 2)
        else:
            print("Gamma > 1.0, Returning 10e10")
            return Scalar[Self.dtype](10e10)

    fn eta(mut self) -> Scalar[Self.dtype]:
        if abs(self.costheta()) < 1.0:
            return -0.5 * log((1.0 - self.costheta()) / (1.0 + self.costheta()))
        else:
            print("eta > 1.0, Returning 10e10")
            return Scalar[Self.dtype](10e10) if self.z() > 0 else -Scalar[Self.dtype](
                10e10
            )

    fn pseudorapidity(mut self) -> Scalar[Self.dtype]:
        return self.eta()

    fn rapidity(mut self) -> Scalar[Self.dtype]:
        return 0.5 * log((self.e() + self.pz()) / (self.e() - self.pz()))

    fn copy(mut self) raises -> Self:
        return Self(self._x, self._y, self._z, self._t)

    fn boostvector(self) raises -> Vector3D[Self.dtype]:
        return Vector3D[Self.dtype](
            self._x / self._t,
            self._y / self._t,
            self._z / self._t,
        )

    fn boost(self, args: Vector3D[Self.dtype]) raises -> Self:
        if len(args) != 3:
            raise Error("Boost vector must be an instance of Vector3D of size 3.")

        var bx: Scalar[Self.dtype] = args[0]
        var by: Scalar[Self.dtype] = args[1]
        var bz: Scalar[Self.dtype] = args[2]

        var b2: Scalar[Self.dtype] = bx**2 + by**2 + bz**2
        var gamma: Scalar[Self.dtype] = 1.0 / sqrt(1.0 - b2)
        var bp: Scalar[Self.dtype] = bx * self.x() + by * self.y() + bz * self.z()
        var gamma2: Scalar[Self.dtype] = 0.0
        if b2 > 0.0:
            gamma2 = (gamma - 1.0) / b2

        var xp: Scalar[Self.dtype] = (
            self.x() + gamma2 * bp * bx - gamma * bx * self.t()
        )
        var yp: Scalar[Self.dtype] = (
            self.y() + gamma2 * bp * by - gamma * by * self.t()
        )
        var zp: Scalar[Self.dtype] = (
            self.z() + gamma2 * bp * bz - gamma * bz * self.t()
        )
        var tp = gamma * (self.t() - bp)

        return Self(xp, yp, zp, tp)

    fn boostplus(self, args: Vector3D[Self.dtype]) raises -> Self:
        return self.boost(args)

    fn boostminus(self, args: Vector3D[Self.dtype]) raises -> Self:
        var bx: Scalar[Self.dtype] = -1.0 * args[0]
        var by: Scalar[Self.dtype] = -1.0 * args[1]
        var bz: Scalar[Self.dtype] = -1.0 * args[2]
        return self.boost(Vector3D[Self.dtype](bx, by, bz))

    fn dot(mut self, other: Self) raises -> Scalar[Self.dtype]:
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
        var boost_vec: Vector3D[Self.dtype] = self.boostvector()
        return self.boostplus(boost_vec)


struct _lorentzvectorIter[
    is_mutable: Bool,
    //,
    lifetime: Origin[mut=is_mutable],
    dtype: DType,
    forward: Bool = True,
](ImplicitlyCopyable):
    """Iterator for LorentzVector.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: LorentzVector[Self.dtype]
    var length: Int

    fn __init__(
        out self,
        array: LorentzVector[Self.dtype],
        length: Int,
    ):
        self.index = 0 if Self.forward else length
        self.length = length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Scalar[Self.dtype]:
        @parameter
        if Self.forward:
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
        if Self.forward:
            return self.index < self.length
        else:
            return self.index > 0

    fn __len__(self) -> Int:
        @parameter
        if Self.forward:
            return self.length - self.index
        else:
            return self.index
