# ===----------------------------------------------------------------------=== #
# This module implements a fixed-size 3-dimensional vector `Vector3D`
# with common vector operations (magnitude, dot, cross, rotation, etc.)
# and convenient accessors for Cartesian and spherical coordinates.
# ===----------------------------------------------------------------------=== #

from sys import simd_width_of
from algorithm import vectorize

from math import sqrt, acos, atan2, sinh, log, sin, cos, tan

from ..constants import pi

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

struct Vector3D[dtype: DType = DType.float64](
    ImplicitlyCopyable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    comptime size: Int = 3
    """The size of the Vector."""

    # Fields
    var _x: Scalar[Self.dtype]
    var _y: Scalar[Self.dtype]
    var _z: Scalar[Self.dtype]
    """3D vector data."""

    # """LIFETIME METHODS"""
    @always_inline("nodebug")
    fn __init__(out self):
        """
        Initializes a 3D vector with zero elements.
        """
        self._x = 0
        self._y = 0
        self._z = 0

    @always_inline("nodebug")
    fn __init__(out self, data: List[Scalar[Self.dtype]]) raises:
        """
        Initializes a 3D vector with the given List of elements.
        """
        if len(data) != self.size - 1:
            raise Error("Length of input should be 3")
        self._x = data[0]
        self._y = data[1]
        self._z = data[2]

    fn __init__(
        out self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
    ):
        """
        Initializes a 3D vector with the given elements.
        """
        self._x = x
        self._y = y
        self._z = z

    fn __copyinit__(out self, other: Self):
        """
        Initializes a 3D vector as a copy of another vector.
        """
        self._x = other._x
        self._y = other._y
        self._z = other._z

    # """GETTER & SETTER METHODS"""
    fn __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        """
        Retrieve a component by index.

        Args:
            index: Index of the component to retrieve. 0 -> x, 1 -> y, 2 -> z.

        Returns:
            The scalar component at the provided index.

        Raises:
            Error: If index is out of the valid range [0, 2].
        """
        if index >= 3:
            raise Error("Invalid index: index exceeds size")
        if index == 0:
            return self._x
        elif index == 1:
            return self._y
        else:
            return self._z

    fn __getattr__[name: StringLiteral](self) raises -> Scalar[Self.dtype]:
        """
        Gets vector attributes by name (x, y, z, r, rho, theta, phi).

        This method supports accessing vector components and computed properties via attribute syntax.

        Returns:
            The scalar value of the requested attribute.

        Raises:
            Error: If the attribute name is not recognized.

        Notes:
            Supported attribute names: x, y, z (Cartesian), r (magnitude),
            rho (cylindrical radial), theta and phi (spherical angles).
        """
        if name == "x":
            return self._x
        elif name == "y":
            return self._y
        elif name == "z":
            return self._z
        elif name == "r":
            return self.mag()
        elif name == "rho":
            return self.rho()
        elif name == "theta":
            return self.theta()
        elif name == "phi":
            return self.phi()
        else:
            raise Error(
                "AttributeError: 'Vector3D' object has no attribute '"
                + name
                + "'"
            )

    fn __setitem__(mut self, index: Int, value: Scalar[Self.dtype]) raises:
        """
        Set the component at the given index.

        Args:
            index: Index of the component to set. 0 -> x, 1 -> y, 2 -> z.
            value: New scalar value.

        Raises:
            Error: If index is out of the valid range [0, 2].
        """
        if index >= 3:
            raise Error("Invalid index: index exceeds size")
        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value
        else:
            self._z = value

    # """TRAITS"""
    fn __str__(self) -> String:
        """
        To print 3D vector.
        """
        return String.write(self)

    # TODO: remove string allocs by writing to writer directly.
    fn write_to[W: Writer](self, mut writer: W):
        """
        Writes the Vector3D to a writer in a formatted string representation.

        Args:
            writer: The writer object to write to.
        """
        try:
            var printStr: String = "Vector3D: ["
            for i in range(self.size - 1):
                printStr += String(self[i])
                if i != 2:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + String(Self.dtype)
            writer.write(printStr)
        except e:
            writer.write("Cannot convert array to string")

    fn print(self) raises -> None:
        """Prints the Vector3D."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(self) -> String:
        """
        Computes the "official" string representation of Vector3D.

        Returns:
            A string representation of the Vector3D with all components and dtype.
        """
        return (
            "Vector3D[DType."
            + String(Self.dtype)
            + "](x="
            + String(self._x)
            + ", y="
            + String(self._y)
            + ", z="
            + String(self._z)
            + ")"
        )

    fn __len__(self) -> Int:
        """
        Returns the length of the Vector3D.

        Returns:
            The size of the vector (3).
        """
        return self.size - 1

    fn __iter__(self) raises -> _vector3DIter[origin_of(self), Self.dtype]:
        """
        Creates an iterator over elements of the Vector3D.

        Returns:
            An iterator of Vector3D elements, returning copied values.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _vector3DIter[origin_of(self), Self.dtype](
            array=self,
            length=self.size - 1,
        )

    fn __reversed__(
        self,
    ) raises -> _vector3DIter[origin_of(self), Self.dtype, forward=False]:
        """
        Creates a reversed iterator over elements of the Vector3D.

        Returns:
            A reversed iterator of Vector3D elements, returning copied values.
        """

        return _vector3DIter[origin_of(self), Self.dtype, forward=False](
            array=self,
            length=self.size - 1,
        )

    fn typeof(mut self) -> DType:
        """
        Returns the data type of the vector's components.

        Returns:
            The DType of the vector.
        """
        return Self.dtype

    fn typeof_str(mut self) -> String:
        """
        Returns the string representation of the vector's data type.

        Returns:
            A string describing the DType of the vector.
        """
        return Self.dtype.__str__()

    # """COMPARISONS"""
    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Vector3D[DType.bool]:
        """
        Component-wise equality comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise equality results.
        """
        return Vector3D[DType.bool](
            self._x == other._x, self._y == other._y, self._z == other._z
        )

    @always_inline("nodebug")
    fn __invert__(
        self,
    ) raises -> Vector3D[DType.bool] where Self.dtype == DType.bool:
        """
        Itemwise logical NOT (for boolean vectors).

        Returns:
            A new Vector3D[DType.bool] with inverted (negated) boolean values.
        """
        return Vector3D[DType.bool](not self._x, not self._y, not self._z)

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise equality comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise equality results.
        """
        return Vector3D[DType.bool](
            self._x == other, self._y == other, self._z == other
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Vector3D[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise inequality comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise inequality results.
        """
        return ~self.__eq__(other)

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise inequality comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise inequality results.
        """
        return ~self.__eq__(other)

    @always_inline("nodebug")
    fn __lt__(self, other: Vector3D[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise less-than comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise less-than results.
        """
        return Vector3D[DType.bool](
            self._x < other._x, self._y < other._y, self._z < other._z
        )

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise less-than comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise less-than results.
        """
        return Vector3D[DType.bool](
            self._x < other, self._y < other, self._z < other
        )

    @always_inline("nodebug")
    fn __le__(self, other: Vector3D[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise less-than-or-equal comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise less-than-or-equal results.
        """
        return Vector3D[DType.bool](
            self._x <= other._x, self._y <= other._y, self._z <= other._z
        )

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise less-than-or-equal comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise less-than-or-equal results.
        """
        return Vector3D[DType.bool](
            self._x <= other, self._y <= other, self._z <= other
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Vector3D[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise greater-than comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise greater-than results.
        """
        return Vector3D[DType.bool](
            self._x > other._x, self._y > other._y, self._z > other._z
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise greater-than comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise greater-than results.
        """
        return Vector3D[DType.bool](
            self._x > other, self._y > other, self._z > other
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Vector3D[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise greater-than-or-equal comparison.

        Args:
            other: The other Vector3D to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise greater-than-or-equal results.
        """
        return Vector3D[DType.bool](
            self._x >= other._x, self._y >= other._y, self._z >= other._z
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[Self.dtype]) raises -> Vector3D[DType.bool]:
        """
        Component-wise greater-than-or-equal comparison with a scalar.

        Args:
            other: The scalar value to compare with.

        Returns:
            A Vector3D[DType.bool] with element-wise greater-than-or-equal results.
        """
        return Vector3D[DType.bool](
            self._x >= other, self._y >= other, self._z >= other
        )

    # """ARITHMETIC"""
    fn __pos__(self) raises -> Self:
        """
        Unary positive operator.

        Returns:
            A copy of the vector (positive operation).
        """
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative operator.

        Returns:
            A new Vector3D with all components negated.
        """
        return self * Scalar[Self.dtype](-1)

    fn __add__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Add a scalar to each component.

        Args:
            other: Scalar value to add.

        Returns:
            A new Vector3D with each component increased by `other`.
        """
        return Self(
            self._x + other,
            self._y + other,
            self._z + other,
        )

    fn __add__(self, other: Self) -> Self:
        """
        Component-wise addition with another vector.

        Args:
            other: The other Vector3D to add.

        Returns:
            A new Vector3D representing the component-wise sum.
        """
        return Self(
            self._x + other._x,
            self._y + other._y,
            self._z + other._z,
        )

    fn __radd__(mut self, other: Scalar[Self.dtype]) -> Self:
        """
        Right addition with a scalar (scalar + vector).

        Args:
            other: Scalar value to add.

        Returns:
            A new Vector3D with each component increased by `other`.
        """
        return self + other

    fn __radd__(self, other: Self) -> Self:
        """
        Right addition with another vector.

        Args:
            other: The other Vector3D to add.

        Returns:
            A new Vector3D representing the component-wise sum.
        """
        return self + other

    fn __iadd__(mut self, other: Scalar[Self.dtype]):
        """
        In-place addition with a scalar.

        Args:
            other: Scalar value to add to each component.
        """
        self = self + other

    fn __iadd__(mut self, other: Self):
        """
        In-place component-wise addition with another vector.

        Args:
            other: The other Vector3D to add.
        """
        self = self + other

    fn __sub__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Subtract a scalar from each component.

        Args:
            other: Scalar value to subtract.

        Returns:
            A new Vector3D with each component decreased by `other`.
        """
        return Self(
            self._x - other,
            self._y - other,
            self._z - other,
        )

    fn __sub__(self, other: Self) -> Self:
        """
        Component-wise subtraction with another vector.

        Args:
            other: The other Vector3D to subtract.

        Returns:
            A new Vector3D representing the component-wise difference.
        """
        return Self(
            self._x - other._x,
            self._y - other._y,
            self._z - other._z,
        )

    fn __rsub__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Right subtraction with a scalar (scalar - vector).

        Args:
            other: Scalar value to subtract from.

        Returns:
            A new Vector3D representing the negated difference.
        """
        return -(self - other)

    fn __rsub__(self, other: Self) raises -> Self:
        """
        Right subtraction with another vector.

        Args:
            other: The other Vector3D to subtract from.

        Returns:
            A new Vector3D representing the negated difference.
        """
        return -(self - other)

    fn __isub__(mut self, other: Scalar[Self.dtype]):
        """
        In-place subtraction with a scalar.

        Args:
            other: Scalar value to subtract from each component.
        """
        self = self - other

    fn __isub__(mut self, other: Self):
        """
        In-place component-wise subtraction with another vector.

        Args:
            other: The other Vector3D to subtract.
        """
        self = self - other

    fn __mul__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Multiply each component by a scalar.

        Args:
            other: Scalar multiplier.

        Returns:
            A new Vector3D scaled by `other`.
        """
        return Self(
            self._x * other,
            self._y * other,
            self._z * other,
        )

    fn __mul__(self, other: Self) -> Self:
        """
        Component-wise multiplication with another vector.

        Args:
            other: The other Vector3D.

        Returns:
            A new Vector3D with component-wise products.
        """
        return Self(
            self._x * other._x,
            self._y * other._y,
            self._z * other._z,
        )

    fn __rmul__(self, other: Scalar[Self.dtype]) -> Self:
        """
        Right multiplication with a scalar (scalar * vector).

        Args:
            other: Scalar multiplier.

        Returns:
            A new Vector3D scaled by `other`.
        """
        return self * other

    fn __rmul__(self, other: Self) -> Self:
        """
        Right multiplication with another vector (component-wise).

        Args:
            other: The other Vector3D.

        Returns:
            A new Vector3D with component-wise products.
        """
        return self * other

    fn __imul__(mut self, other: Scalar[Self.dtype]):
        """
        In-place multiplication with a scalar.

        Args:
            other: Scalar multiplier.
        """
        self = self * other

    fn __imul__(mut self, other: Self):
        """
        In-place component-wise multiplication with another vector.

        Args:
            other: The other Vector3D.
        """
        self = self * other

    fn __pow__(self, p: Int) -> Self:
        """
        Raise each component to a power.

        Args:
            p: The exponent (integer power).

        Returns:
            A new Vector3D with each component raised to power `p`.
        """
        return Self(
            self._x**p,
            self._y**p,
            self._z**p,
        )

    fn __ipow__(mut self, p: Int):
        """
        In-place exponentiation of each component.

        Args:
            p: The exponent (integer power).
        """
        self = self.__pow__(p)

    fn __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Divide each component by a scalar.

        Args:
            other: Scalar divisor.

        Raises:
            Error: If `other` is zero.

        Returns:
            A new Vector3D representing the scaled vector.
        """
        if other == 0:
            raise Error("Division by zero error in Vector3D.__truediv__")
        return Self(
            self._x / other,
            self._y / other,
            self._z / other,
        )

    fn __truediv__(self, other: Self) raises -> Self:
        """
        Component-wise division by another vector.

        Args:
            other: The vector to divide by (component-wise).

        Raises:
            Error: If any component of `other` is zero.

        Returns:
            A new Vector3D representing the component-wise division.
        """
        if other._x == 0 or other._y == 0.0 or other._z == 0:
            raise Error("Division by zero error in Vector3D.__truediv__")
        return Self(
            self._x / other._x,
            self._y / other._y,
            self._z / other._z,
        )

    fn __rtruediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        """
        Right division with a scalar (scalar / vector).

        Args:
            other: Scalar dividend.

        Raises:
            Error: If any component is zero.

        Returns:
            A new Vector3D representing the scaled division.
        """
        return self.__truediv__(other)

    fn __rtruediv__(self, other: Self) raises -> Self:
        """
        Right division with another vector (component-wise).

        Args:
            other: The vector dividend.

        Raises:
            Error: If any component is zero.

        Returns:
            A new Vector3D representing the component-wise division.
        """
        return self.__truediv__(other)

    fn __itruediv__(mut self, other: Scalar[Self.dtype]) raises:
        """
        In-place division by a scalar.

        Args:
            other: Scalar divisor.

        Raises:
            Error: If `other` is zero.
        """
        self = self.__truediv__(other)

    fn __itruediv__(mut self, other: Self) raises:
        """
        In-place component-wise division by another vector.

        Args:
            other: The vector divisor (component-wise).

        Raises:
            Error: If any component of `other` is zero.
        """
        self = self.__truediv__(other)

    fn distance(mut self, other: Vector3D[Self.dtype]) -> Scalar[Self.dtype]:
        """
        Calculates the Euclidean distance between two vectors.

        Args:
            other: The other vector.

        Returns:
            The Euclidean distance between the two vectors.
        """
        return (self - other).mag()

    # """STATIC METHODS"""
    @staticmethod
    fn origin() -> Self:
        """
        Return the origin vector (0, 0, 0).

        Returns:
            A `Vector3D` instance representing the origin.
        """
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(
        x: Scalar[Self.dtype], y: Scalar[Self.dtype], z: Scalar[Self.dtype]
    ) -> Self:
        """
        Create a Vector3D from Cartesian coordinates.

        Args:
            x: The x-component.
            y: The y-component.
            z: The z-component.

        Returns:
            A `Vector3D` instance constructed from the provided components.
        """
        return Self(x=x, y=y, z=z)

    @staticmethod
    fn fromvector(v: Self) raises -> Self:
        """
        Create a Vector3D by copying values from another vector-like object.

        Args:
            v: A vector-like object (e.g., list, Vector3D) with at least three elements.

        Returns:
            A new `Vector3D` containing the components of `v`.
        """
        return Self(v[0], v[1], v[2])

    @staticmethod
    fn fromsphericalcoords(
        r: Scalar[Self.dtype],
        theta: Scalar[Self.dtype],
        phi: Scalar[Self.dtype],
    ) -> Self:
        """
        Create a Vector3D from spherical coordinates.

        Args:
            r: Radial distance from origin.
            theta: Polar angle measured from the positive z-axis.
            phi: Azimuthal angle in the xy-plane from the positive x-axis.

        Returns:
            A `Vector3D` corresponding to the given spherical coordinates.
        """
        var x: Scalar[Self.dtype] = r * sin(theta) * cos(phi)
        var y: Scalar[Self.dtype] = r * sin(theta) * sin(phi)
        var z: Scalar[Self.dtype] = r * cos(theta)
        return Self(x, y, z)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[Self.dtype], phi: Scalar[Self.dtype], z: Scalar[Self.dtype]
    ) -> Self:
        """
        Create a Vector3D from cylindrical coordinates.

        Args:
            rho: Radial distance in the xy-plane.
            phi: Azimuthal angle in the xy-plane from the positive x-axis.
            z: The z-component.

        Returns:
            A `Vector3D` corresponding to the given cylindrical coordinates.
        """
        var x: Scalar[Self.dtype] = rho * cos(phi)
        var y: Scalar[Self.dtype] = rho * sin(phi)
        return Self(x, y, z)

    @staticmethod
    fn fromlist(mut iterable: List[Scalar[Self.dtype]]) raises -> Self:
        """
        Create a Vector3D from an iterable of length 3.

        Args:
            iterable: An iterable (e.g., list/tuple) containing exactly three scalars [x, y, z].

        Returns:
            A `Vector3D` with components taken from `iterable`.

        Raises:
            Error: If `iterable` does not contain exactly three elements.
        """
        if len(iterable) == 3:
            return Self(iterable[0], iterable[1], iterable[2])
        else:
            raise Error("Iterable size does not fit a 3D Vector")

    # """PROPERTIES"""
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

    # TODO: Implement @property decorator
    fn rho(self) -> Scalar[Self.dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self._x**2 + self._y**2)

    fn mag(self) -> Scalar[Self.dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2 + z^2).
        """
        return sqrt(self._x**2 + self._y**2 + self._z**2)

    fn r(mut self) -> Scalar[Self.dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn costheta(self) -> Scalar[Self.dtype]:
        """
        Calculates the cosine of the angle theta between the vector and the z-axis.

        Returns:
            The cosine of angle theta. Returns 1.0 if the vector's magnitude is zero.
        """
        if self.mag() == 0.0:
            return 1.0
        else:
            return self._z / self.mag()

    fn theta(self, degree: Bool = False) -> Scalar[Self.dtype]:
        """
        Calculates the angle theta between the vector and the z-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle theta in radians or degrees.
        """
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / pi.cast[Self.dtype]()
        else:
            return theta

    fn phi(self, degree: Bool = False) -> Scalar[Self.dtype]:
        """
        Calculates the angle phi in the xy-plane from the positive x-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle phi in radians or degrees.
        """
        var phi = atan2(self._y, self._x)
        if degree == True:
            return phi * 180 / pi.cast[Self.dtype]()
        else:
            return phi

    fn set(
        mut self,
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        z: Scalar[Self.dtype],
    ):
        """
        Sets the vector components to the specified values.

        Args:
            x: The new value for the x-component.
            y: The new value for the y-component.
            z: The new value for the z-component.
        """
        self._x = x
        self._y = y
        self._z = z

    fn tolist(self) -> List[Scalar[Self.dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return [Scalar[Self.dtype](self._x), self._y, self._z]

    fn mag2(self) -> Scalar[Self.dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self._x**2 + self._y**2 + self._z**2

    fn __abs__(mut self) -> Scalar[Self.dtype]:
        """
        Calculates the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn unit(self) -> Self:
        """
        Normalizes the vector to a unit vector.

        Returns:
            A new vector with a magnitude of 1, pointing in the same direction as the original vector.
        """
        var mag_temp = self.mag()
        if mag_temp == 1.0:
            return self
        else:
            return Self(
                self._x / mag_temp,
                self._y / mag_temp,
                self._z / mag_temp,
            )

    fn __nonzero__(mut self) -> Bool:
        """
        Checks if the vector is non-zero.

        Returns:
            True if the vector is non-zero, False otherwise.
        """
        return self.mag() != 0.0

    fn __bool__(mut self) -> Bool:
        """
        Converts the vector's non-zero status to a boolean.

        Returns:
            True if the vector is non-zero, False otherwise.
        """
        return self.__nonzero__()

    fn dot(self, other: Self) -> Scalar[Self.dtype]:
        """
        Computes the dot product of this vector with another vector.

        Args:
            other: The other vector to dot with.

        Returns:
            The scalar dot product of the two vectors.
        """
        return self._x * other._x + self._y * other._y + self._z * other._z

    fn cross(self, other: Self) -> Self:
        """
        Computes the cross product of this vector with another vector.

        Args:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return Self(
            self._y * other._z - self._z * other._y,
            self._z * other._x - self._x * other._z,
            self._x * other._y - self._y * other._x,
        )

    fn rotate(self, mut axis: Self, angle: Scalar[Self.dtype]) raises -> Self:
        """
        Rotates this vector around an arbitrary axis by the specified angle.

        Args:
            axis: The axis vector to rotate around (will be normalized).
            angle: The angle in radians to rotate by.

        Returns:
            A new vector that is this vector rotated around the axis.
        """
        var u = axis.unit()
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)
        var one_minus_cos = 1 - cos_theta

        var x_new = (
            (cos_theta + u[0] * u[0] * one_minus_cos) * self._x
            + (u[0] * u[1] * one_minus_cos - u[2] * sin_theta) * self._y
            + (u[0] * u[2] * one_minus_cos + u[1] * sin_theta) * self._z
        )

        var y_new = (
            (u[0] * u[1] * one_minus_cos + u[2] * sin_theta) * self._x
            + (cos_theta + u[1] * u[1] * one_minus_cos) * self._y
            + (u[1] * u[2] * one_minus_cos - u[0] * sin_theta) * self._z
        )

        var z_new = (
            (u[0] * u[2] * one_minus_cos - u[1] * sin_theta) * self._x
            + (u[1] * u[2] * one_minus_cos + u[0] * sin_theta) * self._y
            + (cos_theta + u[2] * u[2] * one_minus_cos) * self._z
        )

        return Self(x_new, y_new, z_new)

    fn rotate_x(mut self, angle: Scalar[Self.dtype]):
        """
        Rotates the vector around the X-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the X-axis.
        """
        var x_new = self._x
        var y_new = self._y * cos(angle) - self._z * sin(angle)
        var z_new = self._y * sin(angle) + self._z * cos(angle)

        self._x = x_new
        self._y = y_new
        self._z = z_new

    fn rotate_y(mut self, angle: Scalar[Self.dtype]):
        """
        Rotates the vector around the Y-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Y-axis.
        """
        var x_new = self._x * cos(angle) + self._z * sin(angle)
        var y_new = self._y
        var z_new = -self._x * sin(angle) + self._z * cos(angle)

        self._x = x_new
        self._y = y_new
        self._z = z_new

    fn rotate_z(mut self, angle: Scalar[Self.dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self._x * cos(angle) - self._y * sin(angle)
        var y_new = self._x * sin(angle) + self._y * cos(angle)
        var z_new = self._z

        self._x = x_new
        self._y = y_new
        self._z = z_new

    fn cos_angle(mut self, mut other: Self) -> Scalar[Self.dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.

        Args:
            other: The other vector with which to compute the cosine of the angle.

        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other) / (self.mag() * other.mag())

    fn angle(mut self, mut other: Self) -> Scalar[Self.dtype]:
        """
        Computes the angle in radians between this vector and another vector.

        Args:
            other: The other vector with which to compute the angle.

        Returns:
            The angle in radians between the two vectors.
        """
        return acos(self.cos_angle(other))

    # maybe I should use isclose function here since it's float
    fn isparallel(mut self, mut other: Self) -> Bool:
        """
        Determines if this vector is parallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are parallel, False otherwise.
        """
        return self.cos_angle(other) == 1.0

    fn isantiparallel(mut self, mut other: Self) -> Bool:
        """
        Determines if this vector is antiparallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are antiparallel, False otherwise.
        """
        return self.cos_angle(other) == -1.0

    fn isperpendicular(mut self, mut other: Self) -> Bool:
        """
        Determines if this vector is perpendicular to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are perpendicular, False otherwise.
        """
        return self.cos_angle(other) == 0.0

    # """ELEMENTWISE / SIMD"""
    fn act[
        function: fn[type: DType, simd_width: Int] (
            SIMD[type, simd_width]
        ) -> SIMD[type, simd_width]
    ](mut self):
        """
        Apply a SIMD-compatible function element-wise to this vector (in-place).

        The argument `function` is a compile-time parameter with signature
        `fn[type: DType, simd_width: Int](SIMD[type, simd_width]) -> SIMD[type, simd_width]`.
        For `simd_width == 1`, the function is equivalent to a scalar function.

        Notes:
            The provided function must be compatible with the underlying `Self.dtype`.
            The vector is modified in-place (components are overwritten).
            For SIMD widths greater than 1, ensure the function manages SIMD lanes appropriately.

        Examples:
            def my_f[type, w](v: SIMD[type, w]) -> SIMD[type, w]:
                return v * v

            # Apply elementwise (assuming `my_f` is accessible in this scope).
            vec.act[my_f]().
        """
        self._x = function[Self.dtype](self._x)
        self._y = function[Self.dtype](self._y)
        self._z = function[Self.dtype](self._z)


struct _vector3DIter[
    is_mutable: Bool,
    //,
    lifetime: Origin[mut=is_mutable],
    dtype: DType,
    forward: Bool = True,
](ImplicitlyCopyable):
    """Iterator for Vector3D.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Vector3D[Self.dtype]
    var length: Int

    fn __init__(
        out self,
        array: Vector3D[Self.dtype],
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
