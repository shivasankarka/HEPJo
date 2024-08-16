from builtin.math import pow
from sys.intrinsics import _mlirtype_is_eq
from algorithm.functional import elementwise
from algorithm import vectorize
from builtin.type_aliases import AnyLifetime

from tensor import Tensor
from .constants import pi
import . math_funcs as mf
from .traits import vectors

"""
TODO:
1) Rewrite the docstrings
2) Add copy, move methods for vector3D
"""
################################################################################################################
####################################### VECTOR 3D ##############################################################
################################################################################################################


@value
struct _vector3DIter[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
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
        inout self,
        array: Vector3D[dtype],
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
struct Vector3D[dtype: DType = DType.float64](
    Intable, CollectionElement, Sized, Stringable, vectors
):
    # Fields
    var data: DTypePointer[dtype]
    """3D vector data."""
    alias size: Int = 3
    """The size of the Vector."""

    """ Default constructor """

    @always_inline("nodebug")
    fn __init__(inout self):
        """
        Initializes a 3D vector with zero elements.
        """
        self.data = DTypePointer[dtype].alloc(self.size)
        memset_zero(self.data, self.size)

    # @always_inline("nodebug")
    # fn __init__[data: Int](inout self) raises:
    #     """
    #     Initializes a 3D vector with the given elements.
    #     """
    #     constrained[data != self.size, "Length of the input should be 3."]()
    #     self.data = DTypePointer[dtype].alloc(self.size)
    #     @parameter
    #     for i in range(self.size):
    #         self.data[i] = data

    @always_inline("nodebug")
    fn __init__(inout self, *data: Scalar[dtype]) raises:
        """
        Initializes a 3D vector with the given elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 3")
        self.data = DTypePointer[dtype].alloc(self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    @always_inline("nodebug")
    fn __init__(inout self, data: List[Scalar[dtype]]) raises:
        """
        Initializes a 3D vector with the given List of elements.
        """
        if len(data) != self.size:
            raise Error("Length of input should be 3")
        self.data = DTypePointer[dtype].alloc(self.size)
        for i in range(self.size):
            self.data[i] = data[i]

    fn __init__(
        inout self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]
    ):
        """
        Initializes a 3D vector with the given elements.
        """
        self.data = DTypePointer[dtype].alloc(self.size)
        self.data[0] = x
        self.data[1] = y
        self.data[2] = z

    fn __init__(inout self, owned data: DTypePointer[dtype]):
        """
        Initializes a 3D vector with the given Pointer.
        """
        self.data = data

    # TODO: I have used a print block and return 0 for invalid index, there's clash between Stringable and raises if I add a raise Error.
    fn __getitem__(self, index: Int) raises -> Scalar[dtype]:
        if index >= 3:
            raise Error("Invalid index: index exceeds size")
        elif index < 0:
            return self.data.load[width=1](index + self.size)
        else:
            return self.data.load[width=1](index)

    fn __setitem__(inout self, index: Int, value: Scalar[dtype]):
        self.data.store[width=1](index, value)

    fn __len__(self) -> Int:
        """Returns the length of the Vector3D (=3)."""
        return self.size

    # TODO: change this to represent the Int version
    fn __int__(self) -> Int:
        return self.size

    fn __str__(self: Vector3D[dtype]) -> String:
        """
        To use Stringable trait and print the array.
        """
        var printStr: String = "Vector3D: ["
        for i in range(self.size):
            try:
                printStr += str(self[i])
            except:
                print("Cannot convery Vector3D to string.")
            if i != 2:
                printStr += " , "

        printStr += "]" + "; "
        printStr += "dtype=" + str(dtype)
        return printStr

    fn print(self) raises -> None:
        """Prints the Vector3D."""
        print(self.__str__() + "\n")
        print()

    fn __repr__(inout self) -> String:
        """Compute the "official" string representation of Vector3D."""
        return (
            "Vector3D[DType."
            + str(dtype)
            + "](x="
            + str(self.data[0])
            + ", y="
            + str(self.data[1])
            + ", z="
            + str(self.data[2])
            + ")"
        )

    fn __iter__(self) raises -> _vector3DIter[__lifetime_of(self), dtype]:
        """Iterate over elements of the Vector3D, returning copied value.

        Returns:
            An iterator of Vector3D elements.

        Notes:
            Need to add lifetimes after the new release.
        """

        return _vector3DIter[__lifetime_of(self), dtype](
            array=self,
            length=self.size,
        )

    fn __reversed__(
        self,
    ) raises -> _vector3DIter[__lifetime_of(self), dtype, forward=False]:
        """Iterate backwards over elements of the Vector3D, returning
        copied value.

        Returns:
            A reversed iterator of Vector3D elements.
        """

        return _vector3DIter[__lifetime_of(self), dtype, forward=False](
            array=self,
            length=self.size,
        )

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
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__eq__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __eq__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise equivelence between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__eq__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ne__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ne__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise nonequivelence.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ne__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__lt__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __lt__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__lt__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__le__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __le__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__le__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__gt__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __gt__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__gt__](
            self.data, other, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: Vector3D[dtype]) raises -> Vector3D[DType.bool]:
        """
        Itemwise less than or equal to between scalar and Array.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_2_vectors[self.size, dtype, SIMD.__ge__](
            self.data, other.data, result.data
        )
        return result

    @always_inline("nodebug")
    fn __ge__(self, other: SIMD[dtype, 1]) raises -> Vector3D[DType.bool]:
        """
        Itemwise greater than or equal to.
        """
        var result: Vector3D[DType.bool] = Vector3D[DType.bool]()
        mf.compare_vector_and_scalar[self.size, dtype, SIMD.__ge__](
            self.data, other, result.data
        )
        return result

    #
    """ARITHMETIC."""

    fn __add__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__add__](
            self.data, other, result.data
        )
        return result

    fn __add__(self, other: Self) -> Self:
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__add__](
            self.data, other.data, result.data
        )
        return result

    fn __radd__(self, other: Scalar[dtype]) -> Self:
        return self + other

    fn __iadd__(inout self, other: Scalar[dtype]):
        self = self + other

    fn __sub__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__sub__](
            self.data, other, result.data
        )
        return result

    fn __sub__(self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__sub__](
        # self.data, other.data
        # )
        var result: Self = Self()
        mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__sub__](
            self.data, other.data, result.data
        )
        return result

    fn __rsub__(self, other: Scalar[dtype]) -> Self:
        return -(self - other)

    fn __isub__(inout self, other: Scalar[dtype]):
        self = self - other

    fn __mul__(self, other: Scalar[dtype]) -> Self:
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__mul__](
            self.data, other, result.data
        )
        return result

    fn __mul__(self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[self.size, dtype, SIMD.__mul__](
        #     self.data, other.data
        # )
        var result: Self = Self()
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
        var result: Self = Self()
        mf.elementwise_scalar_arithmetic[self.size, dtype, SIMD.__truediv__](
            self.data, other, result.data
        )
        return result

    fn __truediv__(inout self, other: Self) -> Self:
        # return mf.elementwise_array_arithmetic[
        #     self.size, dtype, SIMD.__truediv__
        # ](self.data, other.data)
        var result: Self = Self()
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

    # * STATIC METHODS
    @staticmethod
    fn origin() -> Self:
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]) -> Self:
        return Self(x=x, y=y, z=z)

    @staticmethod
    fn fromvector(inout v: Self) raises -> Self:
        return Self(v[0], v[1], v[2])

    @staticmethod
    fn fromsphericalcoords(
        r: Scalar[dtype], theta: Scalar[dtype], phi: Scalar[dtype]
    ) -> Self:
        var x: Scalar[dtype] = r * math.sin(theta) * math.cos(phi)
        var y: Scalar[dtype] = r * math.sin(theta) * math.sin(phi)
        var z: Scalar[dtype] = r * math.cos(theta)
        return Vector3D(x, y, z)

    @staticmethod
    fn fromcylindricalcoodinates(
        rho: Scalar[dtype], phi: Scalar[dtype], z: Scalar[dtype]
    ) -> Self:
        var x: Scalar[dtype] = rho * math.cos(phi)
        var y: Scalar[dtype] = rho * math.sin(phi)
        return Vector3D(x, y, z)

    @staticmethod
    fn fromlist(inout iterable: List[Scalar[dtype]]) raises -> Self:
        if len(iterable) == 3:
            return Self(iterable[0], iterable[1], iterable[2])
        else:
            raise Error("Iterable size does not fit a 3D Vector")

    # * PROPERTIES
    # TODO : Implement @property decorator for x,y,z once available in Mojo
    fn x(inout self, x: Scalar[dtype]):
        """
        Sets the x-component of the vector.

        Args:
            x: The new value for the x-component.
        """
        self.data[0] = x

    fn x(inout self) -> Scalar[dtype]:
        """
        Returns the x-component of the vector.

        Returns:
            The value of the x-component.
        """
        return self.data[0]

    fn y(inout self, y: Scalar[dtype]):
        """
        Sets the y-component of the vector.

        Args:
            y: The new value for the y-component.
        """
        self.data[1] = y

    fn y(inout self) -> Scalar[dtype]:
        """
        Returns the y-component of the vector.

        Returns:
            The value of the y-component.
        """
        return self.data[1]

    fn z(inout self, z: Scalar[dtype]):
        """
        Sets the z-component of the vector.

        Args:
            z: The new value for the z-component.
        """
        self.data[2] = z

    fn z(inout self) -> Scalar[dtype]:
        """
        Returns the z-component of the vector.

        Returns:
            The value of the z-component.
        """
        return self.data[2]

    # TODO: Implement @property decorator
    fn rho(inout self) -> Scalar[dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return math.sqrt(self.x() ** 2 + self.y() ** 2)

    fn mag(inout self) -> Scalar[dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2 + z^2).
        """
        return math.sqrt(
            self.data[0] ** 2 + self.data[1] ** 2 + self.data[2] ** 2
        )

    fn r(inout self) -> Scalar[dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn costheta(inout self) -> Scalar[dtype]:
        """
        Calculates the cosine of the angle theta between the vector and the z-axis.

        Returns:
            The cosine of angle theta. Returns 1.0 if the vector's magnitude is zero.
        """
        if self.mag() == 0.0:
            return 1.0
        else:
            return self.data[2] / self.mag()

    fn theta(inout self, degree: Bool = False) -> Scalar[dtype]:
        """
        Calculates the angle theta between the vector and the z-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle theta in radians or degrees.
        """
        var theta = math.acos(self.costheta())
        if degree == True:
            return theta * 180 / Scalar[dtype](pi)
        else:
            return theta

    fn phi(inout self, degree: Bool = False) -> Scalar[dtype]:
        """
        Calculates the angle phi in the xy-plane from the positive x-axis.

        Args:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle phi in radians or degrees.
        """
        var phi = math.atan2(self.data[1], self.data[0])
        if degree == True:
            return phi * 180 / Scalar[dtype](pi)
        else:
            return phi

    fn set(inout self, x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype]):
        """
        Sets the vector components to the specified values.

        Args:
            x: The new value for the x-component.
            y: The new value for the y-component.
            z: The new value for the z-component.
        """
        self.data[0] = x
        self.data[1] = y
        self.data[2] = z

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return List[Scalar[dtype]](self.data[0], self.data[1], self.data[2])

    fn mag2(self) -> Scalar[dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self.data[0] ** 2 + self.data[1] ** 2 + self.data[2] ** 2

    fn __abs__(inout self) -> Scalar[dtype]:
        """
        Calculates the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn copy(inout self) -> Self:
        """
        Creates a copy of the vector.

        Returns:
            A new instance of the vector with the same components.
        """
        return Self(self.data[0], self.data[1], self.data[2])

    fn unit(inout self) -> Self:
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
                self.data[0] / mag_temp,
                self.data[1] / mag_temp,
                self.data[2] / mag_temp,
            )

    fn __nonzero__(inout self) -> Bool:
        """
        Checks if the vector is non-zero.

        Returns:
            True if the vector is non-zero, False otherwise.
        """
        return self.mag() != 0.0

    fn __bool__(inout self) -> Bool:
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
        return self.__mul__(other)._reduce_sum()

    fn cross(self, other: Self) -> Self:
        """
        Computes the cross product of this vector with another vector.

        Args:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return Self(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        )

    # TODO: Gotta check this function, It returns non sense values for now lol
    # fn rotate(self, inout axis: Self, angle: Scalar[dtype]) raises:

    #     var u = axis.unit()
    #     var cos_theta = math.cos(angle)
    #     var sin_theta = math.sin(angle)

    #     var x_new = self.data[0] * (u[0]*u[0] * (1 - cos_theta) + cos_theta) +
    #                 self.data[1] * (u[0]*u[1] * (1 - cos_theta) - u[2]*sin_theta) +
    #                 self.data[2] * (u[0]*u[2] * (1 - cos_theta) + u[1]*sin_theta)
    #     var y_new = self.data[0] * (u[0]*u[1] * (1 - cos_theta) + u[2]*sin_theta) +
    #                 self.data[1] * (u[1]*u[1] * (1 - cos_theta) + cos_theta) +
    #                 self.data[2] * (u[1]*u[2] * (1 - cos_theta) - u[0]*sin_theta)
    #     var z_new = self.data[0] * (u[0]*u[2] * (1 - cos_theta) - u[1]*sin_theta) +
    #                 self.data[1] * (u[1]*u[2] * (1 - cos_theta) + u[0]*sin_theta) +
    #                 self.data[2] * (u[2]*u[2] * (1 - cos_theta) + cos_theta)

    #     self.data[0] = x_new
    #     self.data[1] = y_new
    #     self.data[2] = z_new

    fn rotate_x(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the X-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the X-axis.
        """
        var x_new = self.data[0]
        var y_new = self.data[1] * math.cos(angle) - self.data[2] * math.sin(
            angle
        )
        var z_new = self.data[1] * math.sin(angle) + self.data[2] * math.cos(
            angle
        )

        self.data[0] = x_new
        self.data[1] = y_new
        self.data[2] = z_new

    fn rotate_y(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Y-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Y-axis.
        """
        var x_new = self.data[0] * math.cos(angle) + self.data[2] * math.sin(
            angle
        )
        var y_new = self.data[1]
        var z_new = -self.data[0] * math.sin(angle) + self.data[2] * math.cos(
            angle
        )

        self.data[0] = x_new
        self.data[1] = y_new
        self.data[2] = z_new

    fn rotate_z(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.

        Args:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self.data[0] * math.cos(angle) - self.data[1] * math.sin(
            angle
        )
        var y_new = self.data[0] * math.sin(angle) + self.data[1] * math.cos(
            angle
        )
        var z_new = self.data[2]

        self.data[0] = x_new
        self.data[1] = y_new
        self.data[2] = z_new

    fn cos_angle(inout self, inout other: Self) -> Scalar[dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.

        Args:
            other: The other vector with which to compute the cosine of the angle.

        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other) / (self.mag() * other.mag())

    fn angle(inout self, inout other: Self) -> Scalar[dtype]:
        """
        Computes the angle in radians between this vector and another vector.

        Args:
            other: The other vector with which to compute the angle.

        Returns:
            The angle in radians between the two vectors.
        """
        return math.acos(self.cos_angle(other))

    fn isparallel(inout self, inout other: Self) -> Bool:
        """
        Determines if this vector is parallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are parallel, False otherwise.
        """
        return self.cos_angle(other) == 1.0

    fn isantiparallel(inout self, inout other: Self) -> Bool:
        """
        Determines if this vector is antiparallel to another vector.

        Args:
            other: The other vector to compare with.

        Returns:
            True if the vectors are antiparallel, False otherwise.
        """
        return self.cos_angle(other) == -1.0

    fn isperpendicular(inout self, inout other: Self) -> Bool:
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
    ](inout self):
        """
        Applies a specified SIMD-compatible function to each element of the vector and returns the modified vector.

        This method acts as a convenient interface to apply a SIMD function across all elements of the vector. The function should take a SIMD type as input and return a SIMD type as output, defining the transformation to be applied to each element. This method internally uses `_elementwise_function_arithmetic` to perform the operation.

        Parameters:
            function: A function that takes a SIMD type and returns a SIMD type, specifying the operation to be performed on each element.

        """
        mf.elementwise_function_arithmetic[self.size, dtype, function](
            self.data
        )

    fn _reduce_sum(self) -> Scalar[dtype]:
        """
        Computes the sum of all elements in the vector using SIMD operations for efficiency.

        This function performs a reduction operation to sum all elements of the vector. It leverages SIMD capabilities to load and add multiple elements simultaneously, which can significantly speed up the operation on large vectors. The result is a scalar value representing the sum of all elements.

        Returns:
            A scalar of type `dtype` representing the sum of all elements in the vector.
        """
        var reduced = Scalar[dtype](0.0)
        alias simd_width: Int = simdwidthof[dtype]()

        @parameter
        fn vectorize_reduce[simd_width: Int](idx: Int) -> None:
            reduced += self.data.load[width=simd_width](idx).reduce_add()

        vectorize[vectorize_reduce, simd_width](self.size)
        return reduced

    fn to_tensor(inout self) raises -> Tensor[dtype]:
        var t = Tensor[dtype](self.size)
        for i in range(self.size):
            t[i] = self[i]
        return t
