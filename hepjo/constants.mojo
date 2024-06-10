
# * CONSTANTS
alias pi:Float32 = 3.141592653
alias e:Float64 = 1.6e-19
alias h:Float64 = 6.62607015e-34
alias c:Float64 = 299792458
alias avogadro:Float64 = 6.02214076e23
alias boltzmann:Float64 = 1.380649e-23

# * Conversions
alias eV_to_J:Float64 = 1.6e-19
alias J_to_eV:Float64 = 1/eV_to_J

# TODO : maybe implement functions that can convert between natural units and SI.

# struct constant[dtype: DType = DType.float64]:
#     var e: Scalar[dtype]
#     var h: Scalar[dtype]
#     var c: Scalar[dtype]
#     var avogadro: Scalar[dtype] 
#     var boltzmann: Scalar[dtype]

#     fn __init__(inout self):
#         e:Float64 = 1.6e-19
#         h:Float64 = 6.62607015e-34
#         c:Float64 = 299792458
#         avogadro:Float64 = 6.02214076e23
#         boltzmann:Float64 = 1.380649e-23 
#         hbarFloat64: 6.626e-34 /(2*pi) 

#     fn pi(inout self) -> Scalar[dtype]:
#         var pi:Float64 = 3.141592653
#         return Scalar[dtype](self.pi)
