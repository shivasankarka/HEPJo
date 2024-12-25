from testing import assert_raises

# * CONSTANTS
alias pi: Float64 = 3.14159265358979323846

# TODO : implement functions that can convert between natural units and SI.
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
