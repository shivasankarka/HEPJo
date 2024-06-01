from builtin.dtype import DType
from tensor import Tensor, TensorShape
import math

from sys.intrinsics import _mlirtype_is_eq
from algorithm.functional import elementwise
from algorithm import vectorize

from .constants import pi

trait vector_trait:
    fn __init__(inout self):
        ...
    fn __copyinit__(inout self, new: Self):
        ...
    fn __moveinit__(inout self, owned existing: Self):
        ...
    fn __getitem__(inout self, index:Int) -> Scalar[dtype]:
        ...
    fn __setitem__(inout self, index:Int, value:Scalar[dtype]):
        ...
    fn __del__(owned self):
        ...
    fn __len__(self) -> Int:
        ...
    fn __int__(self) -> Int:
        ...
    fn __str__(inout self) -> String:
        ...
    fn __repr__(inout self) -> String:
        ...
    fn __pos__(inout self) -> Self:
        ...
    fn __neg__(inout self) -> Self:
        ...
    fn __eq__(self, other: Self) -> Bool:
        ...
    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        ...
    fn __add__(inout self, other:Self) -> Self:
        ...
    fn __radd__(inout self, s: Scalar[dtype])->Self:
        ...
    fn __iadd__(inout self, s: Scalar[dtype]):
        ...
    fn __sub__(inout self, other:Scalar[dtype]) -> Self:
        ...
    fn __sub__(inout self, other:Self) -> Self:
        ...
    fn __rsub__(inout self, s: Scalar[dtype]) -> Self:
        ...
    fn __isub__(inout self, s: Scalar[dtype]):
        ...
    fn __mul__(self, s: Scalar[dtype])->Self:
        ...
    fn __mul__(self, other: Self)->Self:
        ...
    fn __rmul__(self, s: Scalar[dtype])->Self:
        ...
    fn __imul__(inout self, s: Scalar[dtype]):
        ...
    fn __truediv__(self, s: Scalar[dtype])->Self:
        ...
    fn __truediv__(self, other: Self)->Self:
        ...
    fn __rtruediv__(self, s: Scalar[dtype])->Self:
        ...
    fn __itruediv__(inout self, s: Scalar[dtype]):
        ...
    fn __itruediv__(inout self, other: Self):
        ...
    fn __pow__(self, p: Int)->Self:
        ...
    fn __ipow__(inout self, p: Int):
        ...
    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        ...
    
################################################################################################################
####################################### VECTOR 2D ##############################################################
################################################################################################################

struct Vector3D[dtype: DType = DType.float64](
    Intable, CollectionElement, Sized, Stringable
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

    fn __getitem__(self, index:Int) -> Scalar[dtype]:
        return self._ptr.load[width=1](index)

    fn __setitem__(inout self, index:Int, value:Scalar[dtype]):
        self._ptr.store[width=1](index, value)

    fn __del__(owned self):
        self._ptr.free()

    fn __len__(self) -> Int:
        return self._size

    fn __int__(self) -> Int:
        return self._size

    fn __str__(self) -> String:
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

    fn print(self) -> None:
        print(self.__str__() + "\n")
        print()

    fn __repr__(inout self) -> String:
        return "Vector3D(x="+str(self._ptr[0])+", y="+str(self._ptr[1])+", z="+str(self._ptr[2])+")"

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
    
    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[math.add](other)

    fn __add__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.add](other)

    fn __radd__(inout self, s: Scalar[dtype])->Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn _sub__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[math.sub](other)

    fn __sub__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.sub](other)

    # TODO: I don't know why I am getting error here, so do this later.
    # fn __rsub__(inout self, s: Scalar[dtype])->Self:
    #     return -(self - s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self-s

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elementwise_scalar_arithmetic[math.mul](s)

    fn __mul__(self, other: Self)->Self:
        return self._elementwise_array_arithmetic[math.mul](other)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self*s

    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[math.mul](other)._reduce_sum()

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
        return self._elementwise_scalar_arithmetic[math.div](s)

    fn __truediv__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.div](other)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other:Self):
        self = self.__truediv__(other)

    fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)

    # * STATIC METHODS
    @staticmethod
    fn origin() -> Self:
        # Class method to create an origin vector
        return Self(0.0, 0.0, 0.0)

    @staticmethod
    fn frompoint(x:Scalar[dtype], y:Scalar[dtype], z:Scalar[dtype]) -> Self:
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
    fn fromlist(inout iterable: List[Scalar[dtype]]) -> Optional[Self]:
        if len(iterable) == 3:
            return Self(iterable[0], iterable[1], iterable[2])
        else: # TODO: mayeb implement errors properly using inbulit error class
            print("Error: Length of iterable must be 3")
            return None

    # * PROPERTIES
    # TODO : Implement @property decorator for x,y,z once available in Mojo
    fn x(inout self, x:Scalar[dtype]):
        """
        Sets the x-component of the vector.

        Parameters:
            x: The new value for the x-component.
        """
        self._ptr[0] = x

    fn x(inout self) -> Scalar[dtype]:
        """
        Returns the x-component of the vector.

        Returns:
            The value of the x-component.
        """
        return self._ptr[0]

    fn y(inout self, y:Scalar[dtype]):
        """
        Sets the y-component of the vector.

        Parameters:
            y: The new value for the y-component.
        """
        self._ptr[1] = y

    fn y(inout self) -> Scalar[dtype]:
        """
        Returns the y-component of the vector.

        Returns:
            The value of the y-component.
        """
        return self._ptr[1]

    fn z(inout self, z:Scalar[dtype]):
        """
        Sets the z-component of the vector.

        Parameters:
            z: The new value for the z-component.
        """
        self._ptr[2] = z

    fn z(inout self) -> Scalar[dtype]:
        """
        Returns the z-component of the vector.

        Returns:
            The value of the z-component.
        """
        return self._ptr[2]

    # TODO: Implement @property decorator
    fn rho(inout self) -> Scalar[dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self.x()**2 + self.y()**2)

    fn mag(inout self) -> Scalar[dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2 + z^2).
        """
        return sqrt(self._ptr[0]**2 + self._ptr[1]**2 + self._ptr[2]**2)

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
            return self._ptr[2]/self.mag()

    fn theta(inout self, degree:Bool=False) -> Scalar[dtype]:
        """
        Calculates the angle theta between the vector and the z-axis.

        Parameters:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle theta in radians or degrees.
        """
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / Scalar[dtype](pi)
        else:
            return theta

    fn phi(inout self, degree:Bool=False) -> Scalar[dtype]:
        """
        Calculates the angle phi in the xy-plane from the positive x-axis.

        Parameters:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle phi in radians or degrees.
        """
        var phi = atan2(self._ptr[1], self._ptr[0])
        if degree == True:
         
    fn set(inout self, x:Scalar[dtype], y:Scalar[dtype], z:Scalar[dtype]):
        """
        Sets the vector components to the specified values.

        Parameters:
            x: The new value for the x-component.
            y: The new value for the y-component.
            z: The new value for the z-component.
        """
        self._ptr[0] = x
        self._ptr[1] = y
        self._ptr[2] = z

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return List[Scalar[dtype]](self._ptr[0], self._ptr[1], self._ptr[2])

    fn mag2(self) -> Scalar[dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self._ptr[0]**2 + self._ptr[1]**2 + self._ptr[2]**2

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
        return Self(self._ptr[0], self._ptr[1], self._ptr[2])

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
            return Self(self._ptr[0]/mag_temp, self._ptr[1]/mag_temp, self._ptr[2]/mag_temp)

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

        Parameters:
            other: The other vector to dot with.

        Returns:
            The scalar dot product of the two vectors.
        """
        return self._elementwise_array_arithmetic[math.mul](other)._reduce_sum()

    fn cross(self, other: Self) -> Self:
        """
        Computes the cross product of this vector with another vector.

        Parameters:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return Self(self._ptr[1]*other._ptr[2] - self._ptr[2]*other._ptr[1],
                    self._ptr[2]*other._ptr[0] - self._ptr[0]*other._ptr[2],
                    self._ptr[0]*other._ptr[1] - self._ptr[1]*other._ptr[0])

    #TODO: Gotta check this function, It returns non sense values for now lol
    fn rotate(self, inout axis: Self, angle: Scalar[dtype]):
        var u = axis.unit()
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)

        var x_new = self._ptr[0] * (u[0]*u[0] * (1 - cos_theta) + cos_theta) +
                    self._ptr[1] * (u[0]*u[1] * (1 - cos_theta) - u[2]*sin_theta) +
                    self._ptr[2] * (u[0]*u[2] * (1 - cos_theta) + u[1]*sin_theta)
        var y_new = self._ptr[0] * (u[0]*u[1] * (1 - cos_theta) + u[2]*sin_theta) +
                    self._ptr[1] * (u[1]*u[1] * (1 - cos_theta) + cos_theta) +
                    self._ptr[2] * (u[1]*u[2] * (1 - cos_theta) - u[0]*sin_theta)
        var z_new = self._ptr[0] * (u[0]*u[2] * (1 - cos_theta) - u[1]*sin_theta) +
                    self._ptr[1] * (u[1]*u[2] * (1 - cos_theta) + u[0]*sin_theta) +
                    self._ptr[2] * (u[2]*u[2] * (1 - cos_theta) + cos_theta)

        self._ptr[0] = x_new
        self._ptr[1] = y_new
        self._ptr[2] = z_new

    fn rotate_x(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the X-axis by the specified angle.
        
        Parameters:
            angle: The angle in radians by which to rotate the vector around the X-axis.
        """
        var x_new = self._ptr[0]
        var y_new = self._ptr[1]*cos(angle) - self._ptr[2]*sin(angle)
        var z_new = self._ptr[1]*sin(angle) + self._ptr[2]*cos(angle)

        self._ptr[0] = x_new
        self._ptr[1] = y_new
        self._ptr[2] = z_new

    fn rotate_y(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Y-axis by the specified angle.
        
        Parameters:
            angle: The angle in radians by which to rotate the vector around the Y-axis.
        """
        var x_new = self._ptr[0]*cos(angle) + self._ptr[2]*sin(angle)
        var y_new = self._ptr[1]
        var z_new = -self._ptr[0]*sin(angle) + self._ptr[2]*cos(angle)

        self._ptr[0] = x_new
        self._ptr[1] = y_new
        self._ptr[2] = z_new

    fn rotate_z(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.
        
        Parameters:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self._ptr[0]*cos(angle) - self._ptr[1]*sin(angle)
        var y_new = self._ptr[0]*sin(angle) + self._ptr[1]*cos(angle)
        var z_new = self._ptr[2]

    fn cos_angle(inout self, inout other:Self) -> Scalar[dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.
        
        Parameters:
            other: The other vector with which to compute the cosine of the angle.
        
        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other)/(self.mag()*other.mag())

    fn angle(inout self, inout other:Self) -> Scalar[dtype]:
        """
        Computes the angle in radians between this vector and another vector.
        
        Parameters:
            other: The other vector with which to compute the angle.
        
        Returns:
            The angle in radians between the two vectors.
        """
        return acos(self.cos_angle(other))

    fn isparallel(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is parallel to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are parallel, False otherwise.
        """
        return self.cos_angle(other) == 1.0

    fn isantiparallel(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is antiparallel to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are antiparallel, False otherwise.
        """
        return self.cos_angle(other) == -1.0

    fn isperpendicular(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is perpendicular to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are perpendicular, False otherwise.
        """
        return self.cos_angle(other) == 0.0

    # VECTORIZED MATH OPERATIONS ON VECTOR3D
    fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        """
        Performs an element-wise scalar arithmetic operation on this vector using SIMD. 
        This function applies a specified arithmetic operation to each element of the vector 
        in conjunction with a scalar value such as 
            self + s
            self - s
            self * s
            self / s
        Parameters:
            func: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
            s: The scalar value to be used in the operation. This scalar is broadcast to match the dimensions of the vector elements.

        Returns:
            A new instance of the vector where each element is the result of applying the arithmetic operation between the scalar `s` and the corresponding element of the original vector.
        """
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self._size)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._ptr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self._size)
        return new_array

    
    fn _elementwise_array_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        """
        Performs an element-wise arithmetic operation between two vectors using SIMD (Single Instruction, Multiple Data) techniques.
        
        This function leverages a provided SIMD-compatible function `func` to perform the specified arithmetic operation on corresponding elements of this vector and another vector `other`.
        
        Parameters:
            func: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
            other: The other vector involved in the arithmetic operation.
        
        Returns:
            A new vector instance where each element is the result of the arithmetic operation performed on corresponding elements of the two input vectors.
        """
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._ptr.store[width=simd_width](index, func[dtype, simd_width](self._ptr.load[width=simd_width](index), other._ptr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._size)
        return new_vec

    fn _elementwise_function_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        """
        Applies a SIMD-compatible function element-wise to this vector.
        
        This function takes a SIMD-compatible function `func` that operates on a single SIMD type and applies it to each element of the vector, effectively transforming each element based on the function's logic.
        
        Parameters:
            func: A function that takes a SIMD type and returns a SIMD type, defining the operation to be performed on each element.
        
        Returns:
            A new vector instance where each element is the result of applying `func` to the corresponding element of the original vector.
        """
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._ptr.store[width=simd_width](index, func[dtype, simd_width](self._ptr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._size)
        return new_vec

    fn act[function: fn[type:DType, simd_width:Int](arg: SIMD[type, simd_width]) -> SIMD[type,simd_width]](inout self) -> Self:
        """
        Applies a specified SIMD-compatible function to each element of the vector and returns the modified vector.
        
        This method acts as a convenient interface to apply a SIMD function across all elements of the vector. The function should take a SIMD type as input and return a SIMD type as output, defining the transformation to be applied to each element. This method internally uses `_elementwise_function_arithmetic` to perform the operation.
        
        Parameters:
            function: A function that takes a SIMD type and returns a SIMD type, specifying the operation to be performed on each element.
        
        Returns:
            A new vector instance where each element has been transformed by the specified function.
        """
        return self._elementwise_function_arithmetic[function]()

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
            reduced[0] += self._ptr.load[width = simd_width](idx).reduce_add()
        vectorize[vectorize_reduce, simd_width](self._size)
        return reduced
    
    fn to_tensor(self) -> Tensor[dtype]:
        var t = Tensor[dtype](self._size)
        for i in range(self._size):
            t[i] = self[i]
        return t

################################################################################################################
####################################### VECTOR 2D ##############################################################
################################################################################################################

struct Vector2D[dtype: DType = DType.float64](
    Intable, CollectionElement, Sized, Stringable
    ):
    var _ptr: DTypePointer[dtype]
    var _size: Int

    # Constructors
    # * I need to figure out how to assign the datatype given by user if possible
    fn __init__(inout self):
        # default constructor
        self._size = 2
        self._ptr =  DTypePointer[dtype].alloc(self._size)
        memset_zero(self._ptr, self._size)

    fn __init__(inout self, *data:Scalar[dtype]):
        self._size = 2

        self._ptr = DTypePointer[dtype].alloc(self._size)
        for i in range(self._size):
            self._ptr[i] = data[i]

    fn __init__(inout self, _ptr: DTypePointer[dtype]):
        self._size = 2
        self._ptr = _ptr

    fn __copyinit__(inout self, new: Self):
        self._size = new._size
        self._ptr = new._ptr

    fn __moveinit__(inout self, owned existing: Self):
        self._size = existing._size
        self._ptr = existing._ptr
        existing._ptr = DTypePointer[dtype]()

    fn __getitem__(self, index:Int) -> Scalar[dtype]:
        return self._ptr.load[width=1](index)

    fn __setitem__(inout self, index:Int, value:Scalar[dtype]):
        self._ptr.store[width=1](index, value)

    fn __del__(owned self):
        self._ptr.free()

    fn __len__(self) -> Int:
        return self._size

    fn __int__(self) -> Int:
        return self._size

    fn __str__(self) -> String:
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

    fn print(self) -> None:
        print(self.__str__() + "\n")
        print()

    fn __repr__(inout self) -> String:
        return "Vector2D(x="+str(self._ptr[0])+", y="+str(self._ptr[1])+")"

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
    
    fn __add__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[math.add](other)

    fn __add__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.add](other)

    fn __radd__(inout self, s: Scalar[dtype])->Self:
        return self + s

    fn __iadd__(inout self, s: Scalar[dtype]):
        self = self + s

    fn _sub__(inout self, other:Scalar[dtype]) -> Self:
        return self._elementwise_scalar_arithmetic[math.sub](other)

    fn __sub__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.sub](other)

    # TODO: I don't know why I am getting error here, so do this later.
    # fn __rsub__(inout self, s: Scalar[dtype])->Self:
    #     return -(self - s)

    fn __isub__(inout self, s: Scalar[dtype]):
        self = self-s

    fn __mul__(self, s: Scalar[dtype])->Self:
        return self._elementwise_scalar_arithmetic[math.mul](s)

    fn __mul__(self, other: Self)->Self:
        return self._elementwise_array_arithmetic[math.mul](other)

    fn __rmul__(self, s: Scalar[dtype])->Self:
        return self*s

    fn __imul__(inout self, s: Scalar[dtype]):
        self = self*s

    fn __matmul__(inout self, other:Self) -> Scalar[dtype]:
        return self._elementwise_array_arithmetic[math.mul](other)._reduce_sum()

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
        return self._elementwise_scalar_arithmetic[math.div](s)

    fn __truediv__(inout self, other:Self) -> Self:
        return self._elementwise_array_arithmetic[math.div](other)

    fn __itruediv__(inout self, s: Scalar[dtype]):
        self = self.__truediv__(s)

    fn __itruediv__(inout self, other:Self):
        self = self.__truediv__(other)

    fn __rtruediv__(inout self, s: Scalar[dtype]) -> Self:
        return self.__truediv__(s)

    # * STATIC METHODS
    @staticmethod
    fn origin() -> Self:
        # Class method to create an origin vector
        return Self(0.0, 0.0)

    @staticmethod
    fn frompoint(x:Scalar[dtype], y:Scalar[dtype]) -> Self:
        return Self(x, y)

    @staticmethod
    fn fromvector(inout v:Self) -> Self:
        return Self(v[0], v[1])

    @staticmethod
    fn fromsphericalcoords(r:Scalar[dtype], phi:Scalar[dtype]) -> Self:
        var x:Scalar[dtype] = r * cos(phi) 
        var y:Scalar[dtype] = r * sin(phi) 
        return Vector3D(x,y)

    @staticmethod
    fn fromcylindricalcoodinates(rho:Scalar[dtype], phi:Scalar[dtype]) -> Self:
        var x:Scalar[dtype] = rho * cos(phi)
        var y:Scalar[dtype] = rho * sin(phi)
        return Vector3D(x,y)

    @staticmethod
    fn fromlist(inout iterable: List[Scalar[dtype]]) -> Optional[Self]:
        if len(iterable) == 2:
            return Self(iterable[0], iterable[1])
        else: # TODO: mayeb implement errors properly using inbulit error class
            print("Error: Length of iterable must be 3")
            return None

    # * PROPERTIES
    # TODO : Implement @property decorator for x,y,z once available in Mojo
    fn x(inout self, x:Scalar[dtype]):
        """
        Sets the x-component of the vector.

        Parameters:
            x: The new value for the x-component.
        """
        self._ptr[0] = x

    fn x(inout self) -> Scalar[dtype]:
        """
        Returns the x-component of the vector.

        Returns:
            The value of the x-component.
        """
        return self._ptr[0]

    fn y(inout self, y:Scalar[dtype]):
        """
        Sets the y-component of the vector.

        Parameters:
            y: The new value for the y-component.
        """
        self._ptr[1] = y

    fn y(inout self) -> Scalar[dtype]:
        """
        Returns the y-component of the vector.

        Returns:
            The value of the y-component.
        """
        return self._ptr[1]

    # TODO: Implement @property decorator
    fn rho(inout self) -> Scalar[dtype]:
        """
        Calculates the radial distance in the xy-plane (rho).

        Returns:
            The radial distance rho, calculated as sqrt(x^2 + y^2).
        """
        return sqrt(self.x()**2 + self.y()**2)

    fn mag(inout self) -> Scalar[dtype]:
        """
        Calculates the magnitude (or length) of the vector.

        Returns:
            The magnitude of the vector, calculated as sqrt(x^2 + y^2 + z^2).
        """
        return sqrt(self._ptr[0]**2 + self._ptr[1]**2)

    fn r(inout self) -> Scalar[dtype]:
        """
        Alias for the magnitude of the vector.

        Returns:
            The magnitude of the vector.
        """
        return self.mag()

    fn phi(inout self, degree:Bool=False) -> Scalar[dtype]:
        """
        Calculates the angle phi in the xy-plane from the positive x-axis.

        Parameters:
            degree: If True, returns the angle in degrees, otherwise in radians.

        Returns:
            The angle phi in radians or degrees.
        """
        var phi = atan2(self._ptr[1], self._ptr[0])
        if degree == True:
         
    fn set(inout self, x:Scalar[dtype], y:Scalar[dtype], z:Scalar[dtype]):
        """
        Sets the vector components to the specified values.

        Parameters:
            x: The new value for the x-component.
            y: The new value for the y-component.
            z: The new value for the z-component.
        """
        self._ptr[0] = x
        self._ptr[1] = y

    fn tolist(self) -> List[Scalar[dtype]]:
        """
        Converts the vector components to a list.

        Returns:
            A list containing the scalar components of the vector.
        """
        return List[Scalar[dtype]](self._ptr[0], self._ptr[1])

    fn mag2(self) -> Scalar[dtype]:
        """
        Calculates the squared magnitude of the vector.

        Returns:
            The squared magnitude of the vector.
        """
        return self._ptr[0]**2 + self._ptr[1]**2 

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
        return Self(self._ptr[0], self._ptr[1])

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
            return Self(self._ptr[0]/mag_temp, self._ptr[1]/mag_temp)

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

        Parameters:
            other: The other vector to dot with.

        Returns:
            The scalar dot product of the two vectors.
        """
        return self._elementwise_array_arithmetic[math.mul](other)._reduce_sum()

    fn cross(self, other: Self) -> Scalar[dtype]:
        """
        Computes the cross product of this vector with another vector.

        Parameters:
            other: The other vector to cross with.

        Returns:
            A new vector that is the cross product of this vector and the other vector.
        """
        return self._ptr[0] * other._ptr[1] . - self._ptr[1] * other._ptr[0]

    #TODO: Gotta check this function, It returns non sense values for now lol
    fn rotate(self, inout axis: Self, angle: Scalar[dtype]):
        var u = axis.unit()
        var cos_theta = cos(angle)
        var sin_theta = sin(angle)

        var x_new = self._ptr[0] * (u[0]*u[0] * (1 - cos_theta) + cos_theta) +
                    self._ptr[1] * (u[0]*u[1] * (1 - cos_theta) - u[2]*sin_theta) +
                    self._ptr[2] * (u[0]*u[2] * (1 - cos_theta) + u[1]*sin_theta)
        var y_new = self._ptr[0] * (u[0]*u[1] * (1 - cos_theta) + u[2]*sin_theta) +
                    self._ptr[1] * (u[1]*u[1] * (1 - cos_theta) + cos_theta) +
                    self._ptr[2] * (u[1]*u[2] * (1 - cos_theta) - u[0]*sin_theta)
        var z_new = self._ptr[0] * (u[0]*u[2] * (1 - cos_theta) - u[1]*sin_theta) +
                    self._ptr[1] * (u[1]*u[2] * (1 - cos_theta) + u[0]*sin_theta) +
                    self._ptr[2] * (u[2]*u[2] * (1 - cos_theta) + cos_theta)

        self._ptr[0] = x_new
        self._ptr[1] = y_new
        self._ptr[2] = z_new

    fn rotate_z(inout self, angle: Scalar[dtype]):
        """
        Rotates the vector around the Z-axis by the specified angle.
        
        Parameters:
            angle: The angle in radians by which to rotate the vector around the Z-axis.
        """
        var x_new = self._ptr[0]*cos(angle) - self._ptr[1]*sin(angle)
        var y_new = self._ptr[0]*sin(angle) + self._ptr[1]*cos(angle)
        self.set(x_new, y_new)

    fn cos_angle(inout self, inout other:Self) -> Scalar[dtype]:
        """
        Computes the cosine of the angle between this vector and another vector.
        
        Parameters:
            other: The other vector with which to compute the cosine of the angle.
        
        Returns:
            The cosine of the angle between the two vectors.
        """
        return self.dot(other)/(self.mag()*other.mag())

    fn angle(inout self, inout other:Self) -> Scalar[dtype]:
        """
        Computes the angle in radians between this vector and another vector.
        
        Parameters:
            other: The other vector with which to compute the angle.
        
        Returns:
            The angle in radians between the two vectors.
        """
        return acos(self.cos_angle(other))

    fn isparallel(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is parallel to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are parallel, False otherwise.
        """
        return self.cos_angle(other) == 1.0

    fn isantiparallel(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is antiparallel to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are antiparallel, False otherwise.
        """
        return self.cos_angle(other) == -1.0

    fn isperpendicular(inout self, inout other:Self) -> Bool:
        """
        Determines if this vector is perpendicular to another vector.
        
        Parameters:
            other: The other vector to compare with.
        
        Returns:
            True if the vectors are perpendicular, False otherwise.
        """
        return self.cos_angle(other) == 0.0

    # VECTORIZED MATH OPERATIONS ON VECTOR3D
    fn _elementwise_scalar_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, s: Scalar[dtype]) -> Self:
        """
        Performs an element-wise scalar arithmetic operation on this vector using SIMD. 
        This function applies a specified arithmetic operation to each element of the vector 
        in conjunction with a scalar value such as 
            self + s
            self - s
            self * s
            self / s
        Parameters:
            func: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
            s: The scalar value to be used in the operation. This scalar is broadcast to match the dimensions of the vector elements.

        Returns:
            A new instance of the vector where each element is the result of applying the arithmetic operation between the scalar `s` and the corresponding element of the original vector.
        """
        alias simd_width: Int = simdwidthof[dtype]()
        var new_array = Self(self._size)
        @parameter
        fn elemwise_vectorize[simd_width: Int](idx: Int) -> None:
            new_array._ptr.store[width=simd_width](idx, func[dtype, simd_width](SIMD[dtype, simd_width](s), self._ptr.load[width=simd_width](idx)))
        vectorize[elemwise_vectorize, simd_width](self._size)
        return new_array

    
    fn _elementwise_array_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, other: Self) -> Self:
        """
        Performs an element-wise arithmetic operation between two vectors using SIMD (Single Instruction, Multiple Data) techniques.
        
        This function leverages a provided SIMD-compatible function `func` to perform the specified arithmetic operation on corresponding elements of this vector and another vector `other`.
        
        Parameters:
            func: A function that specifies the arithmetic operation to be performed. It takes two SIMD arguments and returns a SIMD result.
            other: The other vector involved in the arithmetic operation.
        
        Returns:
            A new vector instance where each element is the result of the arithmetic operation performed on corresponding elements of the two input vectors.
        """
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._ptr.store[width=simd_width](index, func[dtype, simd_width](self._ptr.load[width=simd_width](index), other._ptr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._size)
        return new_vec

    fn _elementwise_function_arithmetic[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        """
        Applies a SIMD-compatible function element-wise to this vector.
        
        This function takes a SIMD-compatible function `func` that operates on a single SIMD type and applies it to each element of the vector, effectively transforming each element based on the function's logic.
        
        Parameters:
            func: A function that takes a SIMD type and returns a SIMD type, defining the operation to be performed on each element.
        
        Returns:
            A new vector instance where each element is the result of applying `func` to the corresponding element of the original vector.
        """
        alias simd_width = simdwidthof[dtype]()
        var new_vec = Self()
        @parameter
        fn vectorized_arithmetic[simd_width:Int](index: Int) -> None:
            new_vec._ptr.store[width=simd_width](index, func[dtype, simd_width](self._ptr.load[width=simd_width](index)))

        vectorize[vectorized_arithmetic, simd_width](self._size)
        return new_vec

    fn act[function: fn[type:DType, simd_width:Int](arg: SIMD[type, simd_width]) -> SIMD[type,simd_width]](inout self) -> Self:
        """
        Applies a specified SIMD-compatible function to each element of the vector and returns the modified vector.
        
        This method acts as a convenient interface to apply a SIMD function across all elements of the vector. The function should take a SIMD type as input and return a SIMD type as output, defining the transformation to be applied to each element. This method internally uses `_elementwise_function_arithmetic` to perform the operation.
        
        Parameters:
            function: A function that takes a SIMD type and returns a SIMD type, specifying the operation to be performed on each element.
        
        Returns:
            A new vector instance where each element has been transformed by the specified function.
        """
        return self._elementwise_function_arithmetic[function]()

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
            reduced[0] += self._ptr.load[width = simd_width](idx).reduce_add()
        vectorize[vectorize_reduce, simd_width](self._size)
        return reduced
    
    fn to_tensor(self) -> Tensor[dtype]:
        var t = Tensor[dtype](self._size)
        for i in range(self._size):
            t[i] = self[i]
        return t

#####################################################################################
#####################################################################################
#####################################################################################