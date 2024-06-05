from builtin.dtype import DType
from tensor import Tensor, TensorShape, rand

# Intable, Stringable
struct array[dtype: DType = DType.float32]():
    var _arr: Tensor[dtype]
    alias simd_width: Int = simdwidthof[dtype]()

    fn __init__(inout self):
        self._arr = Tensor[dtype]()
        
    fn __init__(inout self, shape: TensorShape, random:Bool = False):
        # default constructor
        if random:
            self._arr = rand[dtype](shape)
        else:
            self._arr = Tensor[dtype](shape)

    fn __init__(inout self, shape: TensorShape, values:List[Scalar[dtype]],random:Bool = False):
        # default constructor
        if random:
            self._arr = rand[dtype](shape)
        else:
            self._arr = Tensor[dtype](shape, values)

    fn __init__(inout self, data: Tensor[dtype]):
        self._arr = data

    # fn __init__(inout self, list: List[Scalar[dtype]]):
    #     for i in range(list.__len__()):
    #         self._arr[i] = list[i]

    # fn __getitem__(inout self, index: Int) -> Scalar[dtype]:
    #     return self._arr[index]

    # fn __setitem__(inout self, index: Int, value: Scalar[dtype]):
    #     self._arr[index] = value
    
    fn num_elements(inout self) -> Int:
        return self._arr.num_elements()

    fn shape(inout self) -> TensorShape:
        return self._arr.shape()

    fn __str__(inout self) -> String:
        return self._arr.__str__()

    fn print(inout self):
        print(self.__str__())

    fn __mul__(inout self, other: array[dtype]) raises -> array[dtype]:
        return array[dtype](self._arr * other._arr)

    # fn __matmul__(inout self, other: array[dtype]) -> array[dtype]:
    #     return array[dtype](self._arr@other._arr)

