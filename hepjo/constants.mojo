from testing import assert_raises

# * CONSTANTS
alias pi: Float64 = 3.14159265358979323846

# TODO : maybe implement functions that can convert between natural units and SI.


struct constant[dtype: DType = DType.float32]:
    var e: Scalar[dtype]
    var h: Scalar[dtype]
    var c: Scalar[dtype]
    var avogadro: Scalar[dtype]
    var boltzmann: Scalar[dtype]
    var hbar: Scalar[dtype]

    fn __init__(inout self):
        self.e = 1.6e-19
        self.h = 6.62607015e-34
        self.c = 299792458
        self.avogadro = 6.02214076e23
        self.boltzmann = 1.380649e-23
        self.hbar = 6.626e-34 / (2.0 * pi)


struct UnitConverter:
    var conversion_table: Dict[String, Dict[String, Float64]]

    fn __init__(inout self) raises:
        self.conversion_table = Dict[String, Dict[String, Float64]]()
        self.conversion_table["eV"]["J"] = 1.6e-19
        self.conversion_table["J"]["eV"] = 1 / 1.6e-19

    fn convert(
        inout self, from_unit: String, to_unit: String, value: Float64
    ) raises -> Float64:
        with assert_raises():
            if from_unit == to_unit:
                raise Error("Error: Same input and output unit provided")
        return self.conversion_table[from_unit][to_unit] * value
