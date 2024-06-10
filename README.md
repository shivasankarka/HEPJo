# SciJo: Scientific Computing in Mojo

<img src="extras/SciJo.jpeg" alt="logo" width="250"/>

HEPJo is a library for numerical computations in Mojo, inspired by [Scikit-HEP](https://scikit-hep.org) library in Python. Our future goal is to incorporate as many features from Scikit-HEP as possible. Currently, HEPJo includes functions for numerical operations in particle physics, such as LorentzVector, Vector3D, Vector2D with plans to expand further.

## Motivation and Goals

I've been keeping an eye on Mojo since its debut and have been eager to use it for scientific computations. However, since Mojo is still quite new, it lacks many of the features found in well-known scientific libraries in Python, such as [Numpy](https://numpy.org), [Scipy](https://scipy.org), and [ScikitHEP](https://scikit-hep.org). 

* I mainly started this project to implement features specific to particle physics and as a general replacement for Scikit-HEP library. This project is also a great way for me to get hands-on experience with Mojo. I'm very open to contributions, so if you're interested or have any ideas or suggestions for features, please feel free to open a issue, PR.

* At present, I've established most of the basic functionalities directly in Mojo, utilizing vectorization and parallelization to maximize performance wherever possible.

* LorentzVector and Vector3D are implemented fully and maintains almost the same syntax as Python's Scikit-HEP library. If you have any ideas to implement or suggestions, please feel free to contact me. 

## Features
For a full list of features currently implemented, please check [Available features](Features.md) for a list of function and properties implemented. 

## Installation

Clone the repository and build the mojo package (Currently not all functions are documented and therefore "mojo package" gives errors, I will resolve this soon. Please download and import it as module for now)

## Usage

Since HEPJo is very early in development, there's not much documentation. I will try to write a proper documentation soon. Please go through the code to understand all the available options. Following is a simple example, Please take a look at doc.mojo file for syntax and options available (Only Vector3D for now). You can also go through test.mojo to find syntax and options for now (It's messy and not organized right now).

```
import hepjo as hj
from hepjo import Vector3D, LorentzVector

var vec = hj.Vector3D(1.0,2.0,3.0)
var vec1 = hj.Vector3D(1.0,2.0,3.0)
print(dot(vec, vec1))

var lvec = LorentzVector(1.0, 2.0, 3.0, 10.0) # px, py, pz, e
var restframe_lvec = lvec.torestframe() #returns to rest frame four vector

var lvec1 = LorentzVector.setpxpypzm(3.0, 4.0, 5.0, 10.0)
var boosted_lvec = lvec.boost(lvec1.boostvector()) # Boosts the Lorentz vector lvec using lvec1
```

## Contributions
Anyone interested in contributing or have interesting ideas, your suggestions are always welcomed. please feel free to contact me or open an issue. 

## Updates
Initially, I started developing a general purpose replacement for numpy, scipy, and Scikit-HEP in Mojo. However, I recently discovered another team working on a numpy replacement called [NuMojo](https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo). I have joined their group and will contribute to that project to build a general purpose replacement for numpy, scipy (You can find my current fork [here](https://github.com/shivasankarka/NuMojo)). I will continue working on this project as well, but I intend to focus on creating a specialized Scikit-HEP replacement tailored specifically for high-energy physics. 
