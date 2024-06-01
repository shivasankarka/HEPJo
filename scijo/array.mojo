from builtin.dtype import DType
from tensor import Tensor, TensorShape, rand

from random import rand

# Intable, Stringable
struct array[dtype: DType = DType.float32](
    ):
    var _arr: Tensor[dtype]
    alias simd_width: Int = simdwidthof[dtype]()

    # * I need to figure out how to create data based on user given size
    fn __init__(inout self):
        self._arr = Tensor[dtype]()
        
    fn __init__(inout self, shape: TensorShape, random:Bool = False):
        # default constructor
        if random:
            self._arr = rand[dtype](shape)
        else:
            self._arr = Tensor[dtype](shape)

    fn __init__(inout self, data: Tensor[dtype]):
        self._arr = data

    # fn __init__(inout self, list: List[Scalar[dtype]]):
    #     for i in range(list.__len__()):
    #         self._arr[i] = list[i]

    