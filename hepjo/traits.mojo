#Traits

trait vectors:
    """Default constructor."""
    @always_inline("nodebug")
    fn __init__(inout self):
        """
        Initializes a 3D vector with zero elements.
        """
        pass

    fn __len__(self) -> Int:
        pass

    fn __str__(self) -> String:
        pass

    fn print(self) raises -> None:
        """Prints the Vector3D."""
        pass

    fn typeof(inout self) -> DType:
        pass

    fn typeof_str(inout self) -> String:
        pass
