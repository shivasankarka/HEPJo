from builtin.dtype import DType
from tensor import Tensor, TensorShape

# Intable, Stringable
struct array[T: DType = DType.float32](Stringable):
    var _arr: Tensor[T]
    alias simd_width: Int = simdwidthof[T]()

    fn __init__(inout self):
        self._arr = Tensor[T]()
        
    fn __init__(inout self, shape: TensorShape, random:Bool = False):
        # default constructor
        if random:
            self._arr = rand[T](shape)
        else:
            self._arr = Tensor[T](shape)

    fn __init__(inout self, shape: TensorShape, values:List[Scalar[T]]):
        # default constructor
        self._arr = Tensor[T](shape, values)

    fn __init__(inout self, data: Tensor[T]):
        self._arr = Tensor[T](TensorShape(data.shape()))
        for i in range(data.num_elements()):
            self._arr[i] = data[i]

    fn __init__(inout self, data: array[T]):
        self._arr = data._arr 

    fn __getitem__(inout self, *indices: Int) -> Scalar[T]:
        return self._arr.__getitem__(indices)

    fn __setitem__(inout self, *indices: Int, value: Scalar[T]):
        self._arr.__setitem__(indices, value)
    
    fn num_elements(inout self) -> Int:
        return self._arr.num_elements()

    fn shape(inout self) -> TensorShape:
        return self._arr.shape()

    fn __str__(self:array[T]) -> String:
        return self._arr.__str__()

    fn print(inout self):
        print(self.__str__())

    fn __repr__(inout self):
        print(self.__str__())

    # fn __add__(inout self, other: array[T]) raises -> array[T]:
    #     return Tensor(self._arr + other._arr)

    # fn __mul__(inout self, other: array[T]) raises -> array[T]:
    #     return array[T](self * other._arr)

    # fn __sub__(inout self, other:array[T]) raises -> array[T]:
    #     return array[T](self - other._arr)

    # fn __matmul__(inout self, other: array[T]) -> array[T]:
        # return array[T](self._arr@other._arr)

