# Defaults
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
from ..constants import pi

################################################################################################################
####################################### VECTOR 2D ##############################################################
################################################################################################################


@value
struct Vector2D[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
    # Fields
    var _buf: UnsafePointer[Scalar[dtype]]
    """2D vector data."""
    alias size: Int = 2
    """The size of the Vector."""

    """ LIFETIME METHODS """

    fn __init__(mut self):
        """
        Initializes a 2D vector with zero elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)

    fn __init__(mut self, data: Scalar[dtype]) raises:
        """
        Initializes a 2D vector with the given elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size):
            self._buf[i] = data

    fn __init__(mut self, *data: Scalar[dtype]) raises:
        """
        Initializes a 2D vector with the given elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 2")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size):
            self._buf[i] = data[i]

    fn __init__(mut self, data: List[Scalar[dtype]]) raises:
        """
        Initializes a 2D vector with the given List of elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 2")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size):
            self._buf[i] = data[i]

    fn __init__(mut self, x: Scalar[dtype], y: Scalar[dtype]):
        """
        Initializes a 2D vector with the given elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self._buf[0] = x
        self._buf[1] = y

    fn __init__(mut self, vector: Vector2D[dtype]):
        """
        Initializes a 2D vector with the given elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self._buf[0] = vector._buf[0]
        self._buf[1] = vector._buf[1]

    fn __copyinit__(out self, other: Vector2D[dtype]):
        """
        Initializes a 3D vector as a copy of another vector.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(self._buf, other._buf, self.size)

    fn __moveinit__(mut self, owned other: Vector2D[dtype]):
        """
        Initializes a 3D vector by moving the data from another vector.
        """
        self._buf = other._buf

    fn __del__(owned self):
        self._buf.free()

    """ GETTER & SETTER METHODS """

    fn __getitem__(self, index: Int) raises -> Scalar[dtype]:
        if index >= 2:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            return self._buf[index + self.size]
        else:
            return self._buf[index]

    fn __setitem__(mut self, index: Int, value: Scalar[dtype]) raises:
        if index >= 2:
            raise Error("Invalid index: index exceeds size")
        self._buf[index] = value

    # TRAITS ###
    fn __str__(self) -> String:
        """
        To print the 2D vector.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            var printStr: String = "Vector2D: ["
            for i in range(self.size):
                printStr += str(self[i])
                if i != 1:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + str(dtype)
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
            + str(dtype)
            + "](x="
            + str(self._buf[0])
            + ", y="
            + str(self._buf[1])
            + ")"
        )

    fn __len__(self) -> Int:
        """Returns the length of the Vector2D (=2)."""
        return self.size

    fn __iter__(self) raises -> _vector2DIter[__origin_of(self), dtype]:
        """Iterate over elements of the Vector2D, returning copied value.

        Returns:
            An iterator of Vector2D elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _vector2DIter[__origin_of(self), dtype](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _vector2DIter[__origin_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the Vector2D, returning
        copied value.

        Returns:
            A reversed iterator of Vector2D elements.
        """

        return _vector2DIter[__origin_of(self), dtype, forward=False](
            array=self,
            length=self.size,
        )

    fn typeof(mut self) -> DType:
        return dtype

    fn typeof_str(mut self) -> String:
        return dtype.__str__()

    """COMPARISIONS."""

    @always_inline("nodebug")
    fn __eq__(self, other: Self) raises -> Vector2D[DType.bool]:
        """
        Itemwise equivalence.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__eq__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__eq__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Vector2D[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ne__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ne__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Vector2D[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__lt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__lt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Vector2D[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__le__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__le__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Vector2D[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__gt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__gt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Vector2D[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ge__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[dtype]) raises -> Vector2D[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        var result: Vector2D[DType.bool] = Vector2D[DType.bool]()
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

    # * since "*" already does element wise calculation, I think matmul is redundant for 1D array, but I could use it for dot products
    fn __matmul__(mut self, other: Self) -> Scalar[dtype]:
        var result: Scalar[dtype] = 0.0
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__mul__](
            self._buf, other._buf, result
        )
        return result

    fn distance(self, other: Vector2D[dtype]) -> Scalar[dtype]:
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
    fn frompoint(x: Scalar[dtype], y: Scalar[dtype]) -> Self:
        return Self(x, y)

    @staticmethod
    fn fromvector(v: Self) raises -> Self:
        return Self(v[0], v[1])

    @staticmethod
    fn fromsphericalcoords(r: Scalar[dtype], phi: Scalar[dtype]) -> Self:
        var x: Scalar[dtype] = r * cos(phi)
        var y: Scalar[dtype] = r * sin(phi)
        return Self(x, y)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[dtype], phi: Scalar[dtype]
    ) -> Self:
        var x: Scalar[dtype] = rho * cos(phi)
        var y: Scalar[dtype] = rho * sin(phi)
        return Self(x, y)

    @staticmethod
    fn fromlist(iterable: List[Scalar[dtype]]) raises -> Self:
        if len(iterable) == 2:
            return Self(iterable[0], iterable[1])
        else:
            raise Error("Error: Length of iterable must be 2")

    # * PROPERTIES
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

    # TODO: Implement @property decorator
    fn rho(self) -> Scalar[dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self._buf[0] ** 2 + self._buf[1] ** 2)

    fn mag(self) -> Scalar[dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self._buf[0] ** 2 + self._buf[1] ** 2)

    fn r(self) -> Scalar[dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn phi(self, degree: Bool = False) -> Scalar[dtype]:
        """
        Calculates the angle phi in the xy-plane from the positive x-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle phi in radians or degrees.
        """
        var phi = atan2(self._buf[1], self._buf[0])
        if degree == True:
            return phi * 180 / pi.cast[dtype]()
        else:
            return phi

    fn set(self, x: Scalar[dtype], y: Scalar[dtype]):
        """
        Sets the vector components to the specified values.

        Args:
            x: The new value for the x-component.
            y: The new value for the y-component.
        """
        self._buf[0] = x
        self._buf[1] = y

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return List[Scalar[dtype]](self._buf[0], self._buf[1])

    fn mag2(self) -> Scalar[dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self._buf[0] ** 2 + self._buf[1] ** 2

    fn __abs__(self) -> Scalar[dtype]:
        """
        Calculates the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn copy(self) -> Self:
        """
        Creates a copy of the vector.

        Returns:
            A new instance of the vector with the same components.
        """
        return Self(self._buf[0], self._buf[1])

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
            return Self(self._buf[0] / mag_temp, self._buf[1] / mag_temp)

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

    fn dot(self, other: Self) -> Scalar[dtype]:
        """
        Computes the dot product of this vector with another vector.

        Args:
            other: The other vector to dot with.

        Returns:
            The scalar dot product of the two vectors.
        """
        return (
            self._buf.load[width=2](0) * other._buf.load[width=2](0)
        ).reduce_add()

    fn cross(self, other: Self) -> Scalar[dtype]:
        """
        Computes the cross product of this vector with another vector.

        Args:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return self._buf[0] * other._buf[1] - self._buf[1] * other._buf[0]

    # TODO: Gotta check this function, It returns non sense values for now lol
    fn rotate(self, angle: Scalar[dtype]) -> Self:
        """
        Rotates the vector by the specified angle.

        Args:
            angle: The angle in radians to rotate by.

        Returns:
            A new vector that is this vector rotated by the angle.
        """
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)

        var x_new = self._buf[0] * cos_theta - self._buf[1] * sin_theta
        var y_new = self._buf[0] * sin_theta + self._buf[1] * cos_theta

        return Self(x_new, y_new)

    fn rotate_z(mut self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self._buf[0] * cos(angle) - self._buf[1] * sin(angle)
        var y_new = self._buf[0] * sin(angle) + self._buf[1] * cos(angle)
        self.set(x_new, y_new)

    fn cos_angle(self, other: Self) -> Scalar[dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.

        Args:
            other: The other vector with which to compute the cosine of the angle.

        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other) / (self.mag() * other.mag())

    fn angle(self, other: Self) -> Scalar[dtype]:
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
        for i in range(self.size):
            self._buf[i] = function(self._buf[i])[0]


#####################################################################################


@value
struct _vector2DIter[
    is_mutable: Bool, //,
    lifetime: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for Vector2D.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Vector2D[dtype]
    alias length: Int = 2

    fn __init__(
        mut self,
        array: Vector2D[dtype],
        length: Int,
    ):
        self.index = 0 if forward else length
        self.array = array

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) raises -> Scalar[dtype]:
        @parameter
        if forward:
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
