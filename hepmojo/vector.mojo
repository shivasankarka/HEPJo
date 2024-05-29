from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

from math import sin, cos, sqrt, acos, atan2

from .constants import pi

alias type = DType.float32
alias type1 = Float32

@value
struct Vector3D:
    var __value:Tensor[type]

    # TODO: I need to figure out how to assign the datatype given by user
    fn __init__(inout self):
        # default constructor
        # Initializes a zero vector
        self.__value = Tensor[type](3)
        # print(self.__value) # * works

    fn __init__(inout self, value:Tensor[type]):
        # initializes a vector given by
        self.__value = value
        # print(self.__value) # * works

    fn __init__(inout self, x:type1, y:type1, z:type1):
        # initializes a vector from given points
        self.__value = Tensor[DType.float32](TensorShape(3), List[Float32](x,y,z))
        # print(self.__value)

    fn __copyinit__(inout self, existing: Self):
        self.__value = existing.__value

    # not sure about this rn
    fn __moveinit__(inout self, owned existing: Self):
        self.__value = existing.__value

    fn __getitem__(inout self, index:Int) -> type1:
        return self.__value[index]

    fn __setitem__(inout self, index:Int, value:type1):
        self.__value[index] = value

    fn __repr__(inout self) -> String:
        return "Vector3D({self.__value[0]}, {self.__value[1]}, {self.__value[2]})"

    fn __str__(inout self) -> String:
        return "Vector3D({self.__value[0]}, {self.__value[1]}, {self.__value[2]})"

    # * Are these done in SIMD or not?
    fn __add__(inout self, other:Vector3D) raises -> Vector3D:
        return Vector3D(self.__value + other.__value)

    fn __add__(inout self, other:type1) raises -> Vector3D:
        return Vector3D(self.__value + other)

    fn __sub__(inout self, other:Vector3D) raises -> Vector3D:
        return Vector3D(self.__value - other.__value)

    fn __sub__(inout self, other:type1) raises -> Vector3D:
        return Vector3D(self.__value - other)

    fn __mul__(inout self, other:Vector3D) raises -> Vector3D:
        return Vector3D(self.__value * other.__value)

    fn __mul__(inout self, other:type1) raises -> Vector3D:
        return Vector3D(self.__value * other)

    fn __truediv__(inout self, other:Vector3D) raises -> Vector3D:
        return Vector3D(self.__value / other.__value)

    fn __truediv__(inout self, other:Float32) raises -> Vector3D:
        return Vector3D(self.__value / other)

    fn __pow__(inout self, other:Int) raises -> Vector3D:
        return Vector3D(self.__value**other)

    fn __pos__(inout self) -> Vector3D:
        return self

    fn __neg__(inout self) -> Vector3D:
        return Vector3D(-self.__value[0], -self.__value[1], -self.__value[2])

    fn __rand__(inout self, other:Vector3D) -> Vector3D:
        return Vector3D(self.__value + other.__value)

    @staticmethod
    fn origin() -> Vector3D:
        # Class method to create an origin vector
        var temp = Tensor[DType.float32](TensorShape(3), List[Float32](0.0,0.0,0.0))
        return Vector3D(temp) 

    @staticmethod
    fn frompoint(x:type1, y:type1, z:type1) -> Vector3D:
        return Vector3D(x, y, z)

    @staticmethod
    fn fromvector(v:Vector3D) -> Vector3D:
        return Vector3D(v.__value[0],v.__value[1], v.__value[2])

    @staticmethod
    fn fromsphericalcoords(r:type1, theta:type1, phi:type1) -> Vector3D:
        var x:type1 = r * sin(theta) * cos(phi)
        var y:type1 = r * sin(theta) * sin(phi)
        var z:type1 = r * cos(theta)
        return Vector3D(x,y,z)

    @staticmethod
    fn fromcylindricalcoodinates(rho:type1, phi:type1, z:type1) raises -> Vector3D:
        var x:type1 = rho * cos(phi)
        var y:type1 = rho * sin(phi)
        return Vector3D(x,y,z)

    @staticmethod
    fn fromlist(iterable: List[type1]) raises -> Optional[Vector3D]:
        if len(iterable) == 3:
            var temp = Tensor[type](TensorShape(3), iterable)
            return Vector3D(temp)
        else:
            var err:Error = Error()
            raise err

    # TODO : Implement @property decorator for x,y,z once available in Mojo
    fn x(inout self, x:type1):
        self.__value[0] = x

    fn x(inout self) -> type1:
        return self.__value[0]

    fn y(inout self, y:type1):
        self.__value[1] = y

    fn y(inout self) -> type1:
        return self.__value[1]

    fn z(inout self, z:type1):
        self.__value[2] = z

    fn z(inout self) -> type1:
        return self.__value[2]

    # TODO: Implement @property decorator
    fn rho(inout self) -> type1:
        return sqrt(self.x()**2 + self.y()**2)

    fn mag(inout self) -> type1:
        return sqrt(self.__value[0]**2 + self.__value[1]**2 + self.__value[2]**2)

    fn r(inout self) -> type1:
        return self.mag()

    fn costheta(inout self) -> type1:
        if self.mag() == 0.0:
            return 1.0
        else:
            return self.__value[2]/self.mag()

    fn theta(inout self, degree:Bool=False) -> type1:
        var theta = acos(self.costheta())
        if degree == True:
            return theta * 180 / pi
        else:
            return theta

    fn phi(inout self, degree:Bool=False) -> type1:
        var phi = atan2(self.__value[1], self.__value[0])
        if degree == True:
            return phi * 180 / pi
        else:
            return phi




