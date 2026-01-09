from math import sqrt, acos, atan2, sinh, log, sin, cos, tan

# Modules
from ..constants import pi

################################################################################################################
####################################### VECTOR 2D ##############################################################
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

# TODO: Add where constraints to dtype.
struct Vector2D[dtype: DType = DType.float64](
    Representable, Sized, Stringable, Writable, ImplicitlyCopyable
):
    # Aliases
    comptime size: Int = 2
    """The size of the Vector."""

    # Fields
    var _x: Scalar[Self.dtype]
    """The x-component of the vector."""
    var _y: Scalar[Self.dtype]
    """The y-component of the vector."""
    # """2D vector data."""

    # """LIFETIME METHODS."""
    @always_inline("nodebug")
    fn __init__(out self) :
        """
        Initializes a 2D vector with zero elements.
        """
        self._x = 0
        self._y = 0

    @always_inline("nodebug")
    fn __init__(out self, data: Scalar[Self.dtype]) raises :
        """
        Initializes a 2D vector with the given elements.
        """
        self._x = data
        self._y = data

    @always_inline("nodebug")
    fn __init__(out self, data: List[Scalar[Self.dtype]]) raises :
        """
        Initializes a 2D vector with the given List of elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 2")
        self._x = data[0]
        self._y = data[1]

    @always_inline("nodebug")
    fn __init__(out self, x: Scalar[Self.dtype], y: Scalar[Self.dtype]) :
        """
        Initializes a 2D vector with the given elements.
        """
        self._x = x
        self._y = y

    @always_inline("nodebug")
    fn __init__(out self, vector: Vector2D[Self.dtype]) :
        """
        Initializes a 2D vector with the given elements.
        """
        self._x = vector._x
        self._y = vector._y

    fn __copyinit__(out self, other: Vector2D[Self.dtype]):
        """
        Initializes a 3D vector as a copy of another vector.
        """
        self._x = other._x
        self._y = other._y

    # """GETTER & SETTER METHODS."""
    fn __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        if index >= 2:
            raise Error("Invalid index: index exceeds size")
        if index == 0:
            return self._x
        elif index == 1:
            return self._y

    fn __setitem__(mut self, index: Int, value: Scalar[Self.dtype]) raises:
        if index >= 2:
            raise Error("Invalid index: index exceeds size")
        if index == 0:
            self._x = value
        elif index == 1:
            self._y = value

    ### TRAITS ###
    fn __str__(self) -> String:
        """
        To print the 2D vector.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            var printStr: String = "Vector2D: ["
            for i in range(self.size):
                printStr += String(self[i])
                if i != 1:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + String(Self.dtype)
            writer.write(printStr)
        except e:
            writer.write("Cannot convert array to string")

    fn print(self) raises -> None:
        """Prints the Vector2D."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(self) -> String:
        """Compute the "official" string representation of Vector2D."""
        return (
            "Vector2D[DType."
            + String(Self.dtype)
            + "](x="
            + String(self._x)
            + ", y="
            + String(self._y)
            + ")"
        )

    fn __len__(self) -> Int:
        """Returns the length of the Vector2D (=2)."""
        return self.size

    fn __iter__(self) raises -> _vector2DIter[origin_of(self), Self.dtype]:
        """Iterate over elements of the Vector2D, returning copied value.

        Returns:
            An iterator of Vector2D elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _vector2DIter[origin_of(self), Self.dtype](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _vector2DIter[origin_of(self), Self.dtype, forward=False]:
        """Iterate backwards over elements of the Vector2D, returning
        copied value.

        Returns:
            A reversed iterator of Vector2D elements.
        """

        return _vector2DIter[origin_of(self), Self.dtype, forward=False](
            array=self,
            length=self.size,
        )

    fn typeof(mut self) -> DType:
        return Self.dtype

    fn typeof_str(mut self) -> String:
        return Self.dtype.__str__()

    # """COMPARISIONS."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise equivalence.
        """
        return Vector2D[DType.bool](
            self._x == other._x, self._y == other._y
        )

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x == other, self._y == other
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x != other._x, self._y != other._y
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        return Vector2D[DType.bool](
            self._x != other, self._y != other
        )

    @always_inline("nodebug")
    fn __lt__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x < other._x, self._y < other._y
        )

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than.
        """
        return Vector2D[DType.bool](
            self._x < other, self._y < other
        )

    @always_inline("nodebug")
    fn __le__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x <= other._x, self._y <= other._y
        )

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        return Vector2D[DType.bool](
            self._x <= other, self._y <= other
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x > other._x, self._y > other._y
        )

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than.
        """
        return Vector2D[DType.bool](
            self._x > other, self._y > other
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        return Vector2D[DType.bool](
            self._x >= other._x, self._y >= other._y
        )

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[Self.dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        return Vector2D[DType.bool](
            self._x >= other, self._y >= other
        )

    # """ARITHMETIC."""

    fn __pos__(self) raises -> Self:
        """
        Unary positve returens self unless boolean type.
        """
        return self

    fn __neg__(self) raises -> Self:
        """
        Unary negative returens self unless boolean type.
        """
        return self * Scalar[Self.dtype](-1)

    fn __add__(self, other: Scalar[Self.dtype])  -> Self:
        return Self(
            self._x + other,
            self._y + other,
        )

    fn __add__(self, other: Self)  -> Self:
        return Self(
            self._x + other._x,
            self._y + other._y,
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
        )

    fn __sub__(self, other: Self) -> Self:
        return Self(
            self._x - other._x,
            self._y - other._y,
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
        )

    fn __mul__(self, other: Self) -> Self:
        return Self(
            self._x * other._x,
            self._y * other._y,
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
        return Self(
            self._x ** p,
            self._y ** p,
        )

    fn __ipow__(mut self, p: Int):
        self = self.__pow__(p)

    fn __truediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        if other == 0.0:
            raise Error("Error: Division by zero in Vector2D.__truediv__")
        return Self(
            self._x / other,
            self._y / other,
        )

    fn __truediv__(self, other: Self) raises -> Self:
        if other._x == 0.0 or other._y == 0.0:
            raise Error("Error: Division by zero in Vector2D.__truediv__")
        return Self(
            self._x / other._x,
            self._y / other._y,
        )

    fn __rtruediv__(self, other: Scalar[Self.dtype]) raises -> Self:
        return self.__truediv__(other)

    fn __rtruediv__(self, other: Self) raises-> Self:
        return self.__truediv__(other)

    fn __itruediv__(mut self, other: Scalar[Self.dtype]) raises:
        self = self.__truediv__(other)

    fn __itruediv__(mut self, other: Self) raises:
        self = self.__truediv__(other)

    # * since "*" already does element wise calculation, I think matmul is redundant for 1D array, but I could use it for dot products
    fn __matmul__(mut self, other: Self) -> Scalar[Self.dtype]:
        return self.dot(other)

    fn distance(self, other: Self) -> Scalar[Self.dtype]:
        """
        Calculates the Euclidean distance between two vectors.

        Args:
            other: The other vector.

        Returns:
            The Euclidean distance between the two vectors.
        """
        return (self - other).mag()

    # * STATIC METHODS
    @staticmethod
    fn origin() -> Self:
        return Self(0.0, 0.0)

    @staticmethod
    fn frompoint(x: Scalar[Self.dtype], y: Scalar[Self.dtype]) -> Self:
        return Self(x, y)

    @staticmethod
    fn fromvector(v: Self) raises -> Self:
        return Self(v[0], v[1])

    @staticmethod
    fn fromsphericalcoords(
        r: Scalar[Self.dtype], phi: Scalar[Self.dtype]
    ) -> Self:
        var x: Scalar[Self.dtype] = r * cos(phi)
        var y: Scalar[Self.dtype] = r * sin(phi)
        return Self(x, y)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[Self.dtype], phi: Scalar[Self.dtype]
    ) -> Self:
        var x: Scalar[Self.dtype] = rho * cos(phi)
        var y: Scalar[Self.dtype] = rho * sin(phi)
        return Self(x, y)

    @staticmethod
    fn fromlist(iterable: List[Scalar[Self.dtype]]) raises -> Self:
        if len(iterable) == 2:
            return Self(iterable[0], iterable[1])
        else:
            raise Error("Error: Length of iterable must be 2")

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

    # TODO: Implement @property decorator
    fn rho(self) -> Scalar[Self.dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self._x ** 2 + self._y ** 2)

    fn mag(self) -> Scalar[Self.dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self._x ** 2 + self._y ** 2)

    fn r(self) -> Scalar[Self.dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

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

    fn set(mut self, x: Scalar[Self.dtype], y: Scalar[Self.dtype]):
        """
        Sets the vector components to the specified values.

        Args:
            x: The new value for the x-component.
            y: The new value for the y-component.
        """
        self._x = x
        self._y = y

    fn tolist(self) -> List[Scalar[Self.dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return [Scalar[Self.dtype](self._x), self._y]

    fn mag2(self) -> Scalar[Self.dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self._x ** 2 + self._y ** 2

    fn __abs__(self) -> Scalar[Self.dtype]:
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
            return Self(self._x / mag_temp, self._y / mag_temp)

    fn __nonzero__(self) -> Bool:
        """
        Checks if the vector is non-zero.

        Returns:
            True if the vector is non-zero, False otherwise.
        """
        return self.mag() != 0.0

    fn __bool__(self) -> Bool:
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
        return self._x * other._x + self._y * other._y

    fn cross(self, other: Self) -> Scalar[Self.dtype]:
        """
        Computes the cross product of this vector with another vector.

        Args:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return self._x * other._y - self._y * other._x

    # TODO: Gotta check this function, It returns non sense values for now lol
    fn rotate(self, angle: Scalar[Self.dtype]) -> Self:
        """
        Rotates the vector by the specified angle.

        Args:
            angle: The angle in radians to rotate by.

        Returns:
            A new vector that is this vector rotated by the angle.
        """
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)

        var x_new =  self._x * cos_theta -  self._y * sin_theta
        var y_new =  self._x * sin_theta +  self._y * cos_theta

        return Self(x_new, y_new)

    fn rotate_z(mut self, angle: Scalar[Self.dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new =  self._x * cos(angle) -  self._y * sin(angle)
        var y_new =  self._x * sin(angle) +  self._y * cos(angle)
        self.set(x_new, y_new)

    fn cos_angle(self, other: Self) -> Scalar[Self.dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.

        Args:
            other: The other vector with which to compute the cosine of the angle.

        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other) / (self.mag() * other.mag())

    fn angle(self, other: Self) -> Scalar[Self.dtype]:
        """
        Computes the angle in radians between this vector and another vector.

        Args:
            other: The other vector with which to compute the angle.

        Returns:
            The angle in radians between the two vectors.
        """
        return acos(self.cos_angle(other))

    fn isparallel(self, other: Self) -> Bool:
        """
        Determines if this vector is parallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are parallel, False otherwise.
        """
        return self.cos_angle(other) == 1.0

    fn isantiparallel(self, other: Self) -> Bool:
        """
        Determines if this vector is antiparallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are antiparallel, False otherwise.
        """
        return self.cos_angle(other) == -1.0

    fn act[
        function: fn[type: DType, simd_width: Int] (
            SIMD[type, simd_width]
        ) -> SIMD[type, simd_width]
    ](mut self):
        """
        Applies a specified SIMD-compatible function to each element of the vector and returns the modified vector.

        This method acts as a convenient interface to apply a SIMD function across all elements of the vector. The function should take a SIMD type as input and return a SIMD type as output, defining the transformation to be applied to each element. This method internally uses `_elementwise_function_arithmetic` to perform the operation.

        Parameters:
            function: A function that takes a SIMD type and returns a SIMD type, specifying the operation to be performed on each element.

        """
        self._x = function[Self.dtype, 1](self._x)
        self._y = function[Self.dtype, 1](self._y)


#####################################################################################


struct _vector2DIter[
    is_mutable: Bool,
    //,
    lifetime: Origin[mut=is_mutable],
    dtype: DType,
    forward: Bool = True,
](ImplicitlyCopyable):
    """Iterator for Vector2D.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Vector2D[Self.dtype]
    comptime length: Int = 2

    fn __init__(
        out self,
        array: Vector2D[Self.dtype],
        length: Int,
    ):
        self.index = 0 if Self.forward else length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Scalar[Self.dtype]:
        @parameter
        if Self.forward:
            var current_index = self.index
            self.index += 1
            return self.array[current_index]
        else:
            var current_index = self.index
            self.index -= 1
            return self.array[current_index]

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
