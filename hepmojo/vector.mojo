from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

from math import sin, cos

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
        print(self.__value) # * works

    fn __init__(inout self, x:type1, y:type1, z:type1):
        # initializes a vector from given points
        self.__value = Tensor[DType.float32](TensorShape(3), List[Float32](x,y,z))
        print(self.__value)

    fn __copyinit__(inout self, existing: Self):
        self.__value = existing.__value

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







