# SciJo: Scientific Computing in Mojo

<img src="extras/SciJo.jpeg" alt="logo" width="250"/>

SciJo is a library for numerical computations in Mojo, inspired by the likes of Numpy and Scipy from Python. Our future goal is to incorporate as many features from Numpy and Scipy as possible. Currently, SciJo includes functions for numerical operations in particle physics, such as LorentzVector and Vector3D, with plans to expand further.

## Current state and Goals

I've been keeping an eye on Mojo since its debut and have been eager to use it for scientific computations. However, since Mojo is still quite new, it lacks many of the features found in well-known scientific libraries in Python, such as Numpy, Scipy, and ScikitHEP. I mainly started it to implement features specific to particle physics, but I got down the rabbit hole and I got interested in trying my best to implement as much as possible in Mojo and build a general purpose scientific library. This project is also a great way for me to get hands-on experience with Mojo. I'm very open to contributions, so if you're interested or have any ideas or suggestions for features, please feel free to open a PR.

At present, I've established most of the basic functionalities directly in Mojo, utilizing vectorization and parallelization to maximize performance wherever possible. I'm currently contemplating the idea of integrating Arrays and LorentzVectors with the existing Mojo Tensor type to seamlessly work with the existing tensor package in Mojo instead of implementing them from scratch as done right now, If you have any ideas please feel free to contact me.

## Installation

Clone the repository

## Usage

Since SciJo is very early in development, there's not much documentation. I will try to write a proper documentation soon. Please go through the code to understand all the available options. 

```
import scijo as sj

var vec = sj.Vector3D(1.0,2.0,3.0)
var vec1 = sj.Vector3D(1.0,2.0,3.0)
print(dot(vec, vec1))

```
