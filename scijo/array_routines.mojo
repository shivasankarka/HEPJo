from algorithm import vectorize, parallelize

from tensor import Tensor, TensorShape

###############################################
###############################################
################### ARANGE ####################
fn arange[T:DType](start: Scalar[T], stop: Scalar[T], step: Scalar[T]) -> Tensor[T]:
    """
        Function that computes a series of values starting from "start" to "stop" with given "step" size.

        Parameter:
            T: DType         - datatype of the Tensor

        Args:
            start: Scalar[T] - Start value
            stop: Scalar[T]  - End value
            step: Scalar[T]  - Step size between each element.

        Returns:
            Tensor[T] - Tensor of datatype T with elements ranging from "start" to "stop" incremented with "step".
    """
    var result: Tensor[T] = Tensor[T]()
    # var num: Int = (stop - start) / step
    # for i in range(num):
    #     result[i] = start + step * i
    var i:Int = 0
    while result[i] <= stop:
        result[i] = start + step * i
        i+=1
    return result

##############################################
#################### LINEAR SPACE ############
##############################################
fn linspace[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, endpoint: Bool = True, parallel: Bool = False) -> Tensor[T]:
    """
        Function that computes a series of linearly spaced values starting from "start" to "stop" with given size.

        Parameter:
            T: DType         - datatype of the Tensor

        Args:
            start: Scalar[T] - Start value
            stop: Scalar[T]  - End value
            num: Int  - No of linearly spaced elements 

        Returns:
            Tensor[T] - Tensor of datatype T with elements ranging from "start" to "stop" with num elements. 

    """
    if parallel:
        return _linspace_parallel[T](start, stop, num, endpoint)
    else:
        return _linspace_serial[T](start, stop, num, endpoint)

fn _linspace_serial[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, endpoint: Bool = True) -> Tensor[T]:
    var result: Tensor[T] = Tensor[T](TensorShape(num))

    if endpoint:
        var step: Scalar[T] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = start + step * i

    else:
        var step: Scalar[T] = (stop - start) / num
        for i in range(num):
            result[i] = start + step * i

    return result

fn _linspace_parallel[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, endpoint: Bool = True) -> Tensor[T]:
    var result: Tensor[T] = Tensor[T](TensorShape(num))
    alias nelts = simdwidthof[T]()

    if endpoint:
        var step: Scalar[T] = (stop - start) / (num - 1)
        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            # result.data().store[width=nelts](idx, start + step*idx)
            result[idx] = start + step * idx
        parallelize[parallelized_linspace](num)

    else:
        var step: Scalar[T] = (stop - start) / num
        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            # result.data().store[width=nelts](idx, start + step*idx)
            result[idx] = start + step * idx
        parallelize[parallelized_linspace1](num)

    return result

##############################################
#################### LOGSPACE ################
##############################################
fn logspace[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, endpoint: Bool = True, base:Scalar[T]=10, parallel: Bool = False) -> Tensor[T]:
    if parallel:
        return _logspace_parallel[T](start, stop, num, endpoint, base)
    else:
        return _logspace_serial[T](start, stop, num, endpoint, base)

fn _logspace_serial[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, base:Scalar[T], endpoint: Bool = True) -> Tensor[T]:
    var result: Tensor[T] = Tensor[T](TensorShape(num))

    if endpoint:
        var step: Scalar[T] = (stop - start) / (num - 1)
        for i in range(num):
            result[i] = base**(start + step * i)
    else:
        var step: Scalar[T] = (stop - start) / num
        for i in range(num):
            result[i] = base**(start + step * i)
    return result

fn _logspace_parallel[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, base:Scalar[T], endpoint: Bool = True) -> Tensor[T]:
    var result: Tensor[T] = Tensor[T](TensorShape(num))

    if endpoint:
        var step: Scalar[T] = (stop - start) / (num - 1)
        @parameter
        fn parallelized_linspace(idx: Int) -> None:
            result[idx] = base**(start + step * idx)
        parallelize[parallelized_linspace](num)

    else:
        var step: Scalar[T] = (stop - start) / num
        @parameter
        fn parallelized_linspace1(idx: Int) -> None:
            result[idx] = base**(start + step * idx)
        parallelize[parallelized_linspace1](num)

    return result

# ! Outputs wrong values for Integer type, works fine for float type. 
fn geomspace[T:DType](start: Scalar[T], stop: Scalar[T], num: Int, endpoint: Bool = True) -> Tensor[T]:
    var a:Scalar[T] = start

    if endpoint:
        var result:Tensor[T] = Tensor[T](TensorShape(num))
        var r:Scalar[T] = math.pow(stop/start, 1/(num-1))
        for i in range(num):
            result[i] = a * math.pow[T](r, i)
        return result

    else:
        var result:Tensor[T] = Tensor[T](TensorShape(num))
        var r:Scalar[T] = math.pow(stop/start, 1/(num))
        for i in range(num):
            result[i] = a * math.pow[T](r, i)
        return result
    
fn zeros[T:DType](*shape:Int) -> Tensor[T]:
    var tens_shape:VariadicList[Int] = shape
    return Tensor[T](tens_shape)

fn eye[T:DType](N:Int, M:Int, k:Int=0) -> Tensor[T]:
    var result:Tensor[T] = Tensor[T](N,M)
    var one = Scalar[T](1)
    for i in range(N):
        for j in range(M):
            if i == j:
                # result[i*M + j] = one
                result[VariadicList[Int](i,j)] = one
            else:
                continue
    
    return result

fn identity[T:DType](n:Int) -> Tensor[T]:
    return eye[T](n,n)

fn ones[T:DType](*shape:Int) -> Tensor[T]:
    var tens_shape:VariadicList[Int] = shape
    return Tensor[T](tens_shape, Scalar[T](1))

fn fill[T:DType, fill_value:Scalar[T]](*shape:Int) -> Tensor[T]:
    var tens_shape:VariadicList[Int] = shape
    return Tensor[T](tens_shape, fill_value)

fn fill[T:DType](shape:VariadicList[Int], fill_value:Scalar[T]) -> Tensor[T]:
    return Tensor[T](shape, fill_value)
