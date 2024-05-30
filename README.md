# SciJo: Scientific Computing in Mojo

Welcome to SciJo, a scientific computing library designed for the Mojo programming language. Inspired by the functionality of Python's numpy and scipy, SciJo extends these capabilities with specialized features for particle physics, including support for Lorentz Vectors.

## Features

- **Vector and Matrix Operations**: Perform a wide range of mathematical operations on vectors and matrices.
- **Lorentz Vectors**: Specialized support for operations on Lorentz vectors, crucial for calculations in particle physics.
- **High Performance**: Optimized for speed and efficiency in scientific computations.
- **Mojo Integration**: Seamlessly integrates with the Mojo language's type system and syntax.

## Installation

To install SciJo, clone the repository and build the package using your preferred Mojo build system.

## Usage

```
import scijo as sj

var vec = sj.Vector3D(1.0,2.0,3.0)
var vec1 = sj.Vector3D(1.0,2.0,3.0)
print(dot(vec, vec1))

```
