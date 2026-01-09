# Traits
# If we decide to combine vectors in future with traits.
trait vectors:
    """Default constructor."""

    @always_inline("nodebug")
    fn __init__(out self):
        """
        Initializes a 3D vector with zero elements.
        """
        ...

    fn __len__(self) -> Int:
        ...

    fn __str__(self) -> String:
        ...

    fn print(self) raises -> None:
        """Prints the Vector3D."""
        ...
