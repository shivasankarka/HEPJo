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
####################################### VECTOR 3D ##############################################################
################################################################################################################

# maybe _buf being a SIMD[dtype, 4] is better
struct Vector3D[dtype: DType = DType.float64](
    Stringable, Representable, CollectionElement, Sized, Writable
):
    # Fields
    var _buf: UnsafePointer[Scalar[dtype]]
    """3D vector data."""
    alias size: Int = 4
    """The size of the Vector."""

    """ LIFETIME METHODS """
    @always_inline("nodebug")
    fn __init__(mut self):
        """
        Initializes a 3D vector with zero elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)

    @always_inline("nodebug")
    fn __init__(mut self, *data: Scalar[dtype]) raises:
        """
        Initializes a 3D vector with the given elements.
        """
        if len(data) != self.size - 1:
            raise Error("Length of input should be 3")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size - 1):
            self._buf[i] = data[i]

    @always_inline("nodebug")
    fn __init__(mut self, data: List[Scalar[dtype]]) raises:
        """
        Initializes a 3D vector with the given List of elements.
        """
        if len(data) != self.size - 1:
            raise Error("Length of input should be 3")
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memset_zero(self._buf, self.size)
        for i in range(self.size - 1):
            self._buf[i] = data[i]

    fn __init__(mut self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]):
        """
        Initializes a 3D vector with the given elements.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        self._buf[0] = x
        self._buf[1] = y
        self._buf[2] = z
        self._buf[3] = 0

    fn __copyinit__(out self, other: Vector3D[dtype]):
        """
        Initializes a 3D vector as a copy of another vector.
        """
        self._buf = UnsafePointer[Scalar[dtype]].alloc(self.size)
        memcpy(self._buf, other._buf, self.size)

    fn __moveinit__(mut self, owned other: Vector3D[dtype]):
        """
        Initializes a 3D vector by moving the data from another vector.
        """
        self._buf = other._buf

    fn __del__(owned self):
        self._buf.free()

    """ GETTER & SETTER METHODS """

    fn __getitem__(self, index: Int) raises -> Scalar[dtype]:
        if index >= 3:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            return self._buf.load(index + self.size)
        else:
            return self._buf.load(index)

    fn __setitem__(mut self, index: Int, value: Scalar[dtype]) raises:
        if index >= 3:
            raise Error("Invalid index: index exceeds size")
        self._buf.store(index, value)

    ### TRAITS ###
    fn __str__(self) -> String:
        """
        To print 3D vector.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        try:
            var printStr: String = "Vector3D: ["
            for i in range(self.size - 1):
                printStr += str(self[i])
                if i != 2:
                    printStr += " , "

            printStr += "]" + "\n"
            printStr += "dtype=" + str(dtype)
            writer.write(printStr)
        except e:
            writer.write("Cannot convert array to string")

    fn print(self) raises -> None:
        """Prints the Vector3D."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(self) -> String:
        """Compute the "official" string representation of Vector3D."""
        return (
            "Vector3D[DType."
            + str(dtype)
            + "](x="
            + str(self._buf[0])
            + ", y="
            + str(self._buf[1])
            + ", z="
            + str(self._buf[2])
            + ")"
        )

    fn __len__(self) -> Int:
        """Returns the length of the Vector3D (=3)."""
        return self.size - 1

    fn __iter__(self) raises -> _vector3DIter[__origin_of(self), dtype]:
        """Iterate over elements of the Vector3D, returning copied value.

        Returns:
            An iterator of Vector3D elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _vector3DIter[__origin_of(self), dtype](
            array=self,
            length=self.size - 1,
        )

    fn __reversed__(
        self,
    ) raises -> _vector3DIter[__origin_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the Vector3D, returning
        copied value.

        Returns:
            A reversed iterator of Vector3D elements.
        """

        return _vector3DIter[__origin_of(self), dtype, forward=False](
            array=self,
            length=self.size - 1,
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
    fn __eq__(self, other: Self) raises -> Vector3D[DType.bool]:
        """
        Itemwise equivalence.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__eq__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise equivalence between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__eq__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__ne__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__ne__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__lt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__lt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__le__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__le__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__gt__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__gt__](
            self._buf, other, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size - 1, dtype, SIMD.__ge__](
            self._buf, other._buf, result._buf
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Scalar[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size - 1, dtype, SIMD.__ge__](
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

    fn distance(mut self, other: Vector3D[dtype]) -> Scalar[dtype]:
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
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]) -> Self:
        return Self(x=x, y=y, z=z)

    @staticmethod
    fn fromvector(v: Self) raises -> Self:
        return Self(v[0], v[1], v[2])

    @staticmethod
    fn fromsphericalcoords(
        r: Scalar[dtype], theta: Scalar[dtype], phi: Scalar[dtype]
    ) -> Self:
        var x: Scalar[dtype] = r * sin(theta) * cos(phi)
        var y: Scalar[dtype] = r * sin(theta) * sin(phi)
        var z: Scalar[dtype] = r * cos(theta)
        return Vector3D(x, y, z)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[dtype], phi: Scalar[dtype], z: Scalar[dtype]
    ) -> Self:
        var x: Scalar[dtype] = rho * cos(phi)
        var y: Scalar[dtype] = rho * sin(phi)
        return Vector3D(x, y, z)

    @staticmethod
    fn fromlist(mut iterable: List[Scalar[dtype]]) raises -> Self:
        if len(iterable) == 3:
            return Self(iterable[0], iterable[1], iterable[2])
        else:
            raise Error("Iterable size does not fit a 3D Vector")

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

    # TODO: Implement @property decorator
    fn rho(self) -> Scalar[dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt((self._buf.load[width=2](0) ** 2).reduce_add())

    fn mag(self) -> Scalar[dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2 + z^2).
        """
        return sqrt((self._buf.load[width=4](0) ** 2).reduce_add())

    fn r(mut self) -> Scalar[dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn costheta(mut self) -> Scalar[dtype]:
        """
        Calculates the cosine of the angle theta between the vector and the z-axis.

        Returns:
            The cosine of angle theta. Returns 1.0 if the vector's magnitude is zero.
        """
        if self.mag() == 0.0:
            return 1.0
        else:
            return self._buf[2] / self.mag()

    fn theta(mut self, degree: Bool = False) -> Scalar[dtype]:
        """
        Calculates the angle theta between the vector and the z-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle theta in radians or degrees.
        """
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / pi.cast[dtype]()
        else:
            return theta

    fn phi(mut self, degree: Bool = False) -> Scalar[dtype]:
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

    fn set(mut self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]):
        """
        Sets the vector components to the specified values.

        Args:
            x: The new value for the x-component.
            y: The new value for the y-component.
            z: The new value for the z-component.
        """
        self._buf[0] = x
        self._buf[1] = y
        self._buf[2] = z

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return List[Scalar[dtype]](self._buf[0], self._buf[1], self._buf[2])

    fn mag2(self) -> Scalar[dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return (self._buf.load[width=4](0) ** 2).reduce_add()

    fn __abs__(mut self) -> Scalar[dtype]:
        """
        Calculates the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn copy(mut self) -> Self:
        """
        Creates a copy of the vector.

        Returns:
            A new instance of the vector with the same components.
        """
        return Self(self._buf[0], self._buf[1], self._buf[2])

    fn unit(mut self) -> Self:
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
                self._buf[0] / mag_temp,
                self._buf[1] / mag_temp,
                self._buf[2] / mag_temp,
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

    fn dot(self, other: Self) -> Scalar[dtype]:
        """
        Computes the dot product of this vector with another vector.

        Args:
            other: The other vector to dot with.

        Returns:
            The scalar dot product of the two vectors.
        """
        return (self.load[width=4](0) * other.load[width=4](0)).reduce_add()

    fn cross(self, other: Self) -> Self:
        """
        Computes the cross product of this vector with another vector.

        Args:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return Self(
            self._buf[1] * other._buf[2] - self._buf[2] * other._buf[1],
            self._buf[2] * other._buf[0] - self._buf[0] * other._buf[2],
            self._buf[0] * other._buf[1] - self._buf[1] * other._buf[0],
        )

    fn rotate(self, mut axis: Self, angle: Scalar[dtype]) raises -> Self:
        """
        Rotates this vector around an arbitrary axis by the specified angle.

        Args:
            axis: The axis vector to rotate around (will be normalized).
            angle: The angle in radians to rotate by.

        Returns:
            A new vector that is this vector rotated around the axis.
        """
        # Normalize the axis vector
        var u = axis.unit()
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)
        var one_minus_cos = 1 - cos_theta

        # Rodrigues rotation formula
        var x_new = (cos_theta + u[0] * u[0] * one_minus_cos) * self._buf[0] + (
            u[0] * u[1] * one_minus_cos - u[2] * sin_theta
        ) * self._buf[1] + (
            u[0] * u[2] * one_minus_cos + u[1] * sin_theta
        ) * self._buf[
            2
        ]

        var y_new = (
            u[0] * u[1] * one_minus_cos + u[2] * sin_theta
        ) * self._buf[0] + (
            cos_theta + u[1] * u[1] * one_minus_cos
        ) * self._buf[
            1
        ] + (
            u[1] * u[2] * one_minus_cos - u[0] * sin_theta
        ) * self._buf[
            2
        ]

        var z_new = (
            u[0] * u[2] * one_minus_cos - u[1] * sin_theta
        ) * self._buf[0] + (
            u[1] * u[2] * one_minus_cos + u[0] * sin_theta
        ) * self._buf[
            1
        ] + (
            cos_theta + u[2] * u[2] * one_minus_cos
        ) * self._buf[
            2
        ]

        return Self(x_new, y_new, z_new)

    fn rotate_x(mut self, angle: Scalar[dtype]):
        """
        Rotates the vector around the X-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the X-axis.
        """
        var x_new = self._buf[0]
        var y_new = self._buf[1] * cos(angle) - self._buf[2] * sin(angle)
        var z_new = self._buf[1] * sin(angle) + self._buf[2] * cos(angle)

        self._buf[0] = x_new
        self._buf[1] = y_new
        self._buf[2] = z_new

    fn rotate_y(mut self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Y-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Y-axis.
        """
        var x_new = self._buf[0] * cos(angle) + self._buf[2] * sin(angle)
        var y_new = self._buf[1]
        var z_new = -self._buf[0] * sin(angle) + self._buf[2] * cos(angle)

        self._buf[0] = x_new
        self._buf[1] = y_new
        self._buf[2] = z_new

    fn rotate_z(mut self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self._buf[0] * cos(angle) - self._buf[1] * sin(angle)
        var y_new = self._buf[0] * sin(angle) + self._buf[1] * cos(angle)
        var z_new = self._buf[2]

        self._buf[0] = x_new
        self._buf[1] = y_new
        self._buf[2] = z_new

    fn cos_angle(mut self, mut other: Self) -> Scalar[dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.

        Args:
            other: The other vector with which to compute the cosine of the angle.

        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other) / (self.mag() * other.mag())

    fn angle(mut self, mut other: Self) -> Scalar[dtype]:
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
        mf.elementwise_function_arithmetic[self.size, dtype, function](
            self._buf
        )


@value
struct _vector3DIter[
    is_mutable: Bool, //,
    lifetime: Origin[is_mutable],
    dtype: DType,
    forward: Bool = True,
]:
    """Iterator for Vector3D.

    Parameters:
        is_mutable: Whether the iterator is mutable.
        lifetime: The lifetime of the underlying NDArray data.
        dtype: The data type of the item.
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var array: Vector3D[dtype]
    var length: Int

    fn __init__(
        mut self,
        array: Vector3D[dtype],
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
