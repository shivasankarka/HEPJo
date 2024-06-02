from builtin.dtype import DType
from tensor import Tensor, TensorShape, rand

# Intable, Stringable
struct array[dtype: DType = DType.float32]():
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

    fn __getitem__(inout self, index: Int) -> Scalar[dtype]:
        return self._arr[index]

    fn __setitem__(inout self, index: Int, value: Scalar[dtype]):
        self._arr[index] = value
    
    fn num_elements(inout self) -> Int:
        return self._arr.num_elements()

    fn shape(inout self) -> TensorShape:
        return self._arr.shape()
