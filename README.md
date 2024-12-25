<a name="readme-top"></a>
<!-- add these later -->
<!-- [![MIT License][license-shield]][] -->

<div align="center">
  <a href="">
    <img src="./extras/hepjo_img.jpeg" alt="Logo" width="350" height="350">
  </a>

  <h1 align="center" style="font-size: 3em; color: white; font-family: 'Avenir'; text-shadow: 1px 1px orange;">HEPJo</h1>

  <p align="center">
    HEPJo is a high-performance library for numerical computations in particle physics, written in Mojo 🔥 similar to Scikit-HEP in Python.
    <br />
    <br />
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#features">Features</a>
    </li>
    <li>
      <a href="#quick-start">Quick Start</a>
    </li>
    <li>
      <a href="#api-reference">API Reference</a>
    </li>
  </ol>
</details>

## About

HEPJo is a high-performance library for numerical computations in particle physics, written in Mojo. Inspired by [Scikit-HEP](https://scikit-hep.org), it aims to provide fast and efficient implementations of common particle physics operations. The library currently includes implementations of LorentzVector, Vector3D, and Vector2D classes with plans to expand further.

## Installation

1. Clone the repository.
2. Build the package using `mojo package hepjo`
3. Move the hepjo.mojopkg into the directory containing the your code.

## Features

- **High Performance**: Written in Mojo for maximum speed and efficiency
- **Vector Operations**: Complete implementation of 2D, 3D, and 4D (Lorentz) vectors
- **Physics Operations**: Common particle physics calculations including boosts and transformations
- **Type Safety**: Strong typing and compile-time checks
- **Scikit-HEP like Interface**: Familiar API for physics computations

## Quick Start

HEPJo is currently under development and does not yet have comprehensive documentation. Please refer to the tests and example code to explore the full range of functionalities available.

```mojo
import hepjo as hj
from hepjo import Vector3D, LorentzVector

# Create and manipulate 3D vectors
var vec = hj.Vector3D(1.0, 2.0, 3.0)
var vec1 = hj.Vector3D(1.0, 2.0, 3.0)
print(vec.dot(vec1), vec @ vec1)  # Both computes dot product

# Work with Lorentz vectors
var lvec = LorentzVector(1.0, 2.0, 3.0, 10.0)  # px, py, pz, e
var restframe_lvec = lvec.torestframe()  # Transform to rest frame

# Create Lorentz vector with given mass
var lvec1 = LorentzVector.setpxpypzm(3.0, 4.0, 5.0, 10.0)
var boosted_lvec = lvec.boost(lvec1.boostvector())  # Apply boost transformation
```

## API Reference

### Vector2D

The `Vector2D[dtype]` class represents a two-dimensional vector.

#### Methods

- `__init__(x: Scalar[dtype], y: Scalar[dtype])`: Initialize a 3D vector
- `dot(other: Vector2D) -> Float64`: Compute dot product
- `cross(other: Vector2D) -> Vector3D`: Compute cross product
- `mag() -> Scalar[dtype]`: Get vector magnitude
- `mag2() -> Scalar[dtype]`: Get squared magnitude
- `unit() -> Vector2D`: Get unit vector

### Vector3D

The `Vector3D[dtype]` class represents a three-dimensional vector with operations commonly used in particle physics.

#### Methods

- `__init__(x: Scalar[dtype], y: Scalar[dtype], z: Scalar[dtype])`: Initialize a 3D vector
- `dot(other: Vector3D) -> Float64`: Compute dot product
- `cross(other: Vector3D) -> Vector3D`: Compute cross product
- `mag() -> Scalar[dtype]`: Get vector magnitude
- `mag2() -> Scalar[dtype]`: Get squared magnitude
- `unit() -> Vector3D`: Get unit vector

### LorentzVector

The `LorentzVector[dtype]` class represents a four-momentum vector with operations for relativistic calculations.

#### Methods

- `__init__(px: Scalar[dtype], py: Scalar[dtype], pz: Scalar[dtype], e: Scalar[dtype])`: Initialize with momentum components and energy
- `setpxpypzm(px: Scalar[dtype], py: Scalar[dtype], pz: Scalar[dtype], m: Scalar[dtype]) -> LorentzVector`: Create from momentum and mass
- `torestframe() -> LorentzVector`: Transform to rest frame
- `boost(beta: Vector3D) -> LorentzVector`: Apply Lorentz boost
- `mass() -> Scalar[dtype]`: Get invariant mass
- `mass2() -> Scalar[dtype]`: Get squared invariant mass
- `pt() -> Scalar[dtype]`: Get transverse momentum
- `eta() -> Scalar[dtype]`: Get pseudorapidity
- `phi() -> Scalar[dtype]`: Get azimuthal angle
- `boostvector() -> Vector3D`: Get boost vector

## Examples

### Computing Invariant Mass

```python
from hepjo import LorentzVector

# Create two particles
var particle1 = LorentzVector.setpxpypzm(10.0, 20.0, 30.0, 0.139)  # pion
var particle2 = LorentzVector.setpxpypzm(-5.0, 15.0, 25.0, 0.139)  # pion

# Compute invariant mass of the system
var system = particle1 + particle2
print(system.mass())  # Print invariant mass
```

### Boost to Center of Mass Frame

```mojo
from hepjo import LorentzVector

# Create a particle system
var particle = LorentzVector(5.0, 0.0, 0.0, 10.0) # px, py, pz, e
var lab_frame = LorentzVector(0.0, 0.0, 8.0, 12.0) 

# Boost to center of mass frame
var cm_frame = particle.boost(lab_frame.boostvector())
print(cm_frame.p())  # Print momentum in CM frame
```

## Contributing

Contributions are welcome! Please feel free to report issues and submit a Pull Request.

## License

Distributed under the Apache 2.0 License with LLVM Exceptions. See [LICENSE](https://github.com/shivasankarka/HEPJo/blob/main/LICENSE) and the LLVM [License](https://llvm.org/LICENSE.txt) for more information.

## Future Plans
- Integration with PDG particle data.
- Implement a System of Units similar to GEANT4.
- Integration with common HEP analysis frameworks.
- GPU support for large array computations.
- Add more Mojo backends such as [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo).

## Citation
If you find HEPJo useful in your research adventures, consider giving it a shout-out in your citations!

```bibtex
@software{hepjo,
  author = {ShivaSankar K.A},
  title = {HEPJo: High Performance Particle Physics in Mojo},
  year = {2024},
  url = {https://github.com/shivasankarka/hepjo}
}
```