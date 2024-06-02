# TODO

Following are my thoughts, and ideas left to implement for now for each of the mojo file. I will add more soon as I continue to update. Right now, I am mostly interested in implementing basic functionalities in the style of numpy, scipy and scikithep rather than organizing it properly, so the files might be a bit confusing right now. But I plan to organize these soon in future with more options for user.

# statistics
* Implement axis parameter like numpy to act the functions only on specific axis of N-D tensor

# array
* I am not sure if a separate SciJo specific array type would be better or not, nevertheless it's partially implemented (both from scratch and as a wrapper for Tensor type)

# constants
* I think defining all constants in scikithep style with struct would be better for Floating point precision

# Functions calculus
* Only trapz is implemented, I should consider what other important functions to implement

# Vector
* Vector3D, Vector2D are implemented fully. There are few minor tweak here and there as mentioned in commments

# LorentzVector
* LorentzVecotr is implemented fully. There are few minor tweak here and there as mentioned in commments

# Interpolate
* Implemented linear, quadratic, cubic 1D interpolations similar to np.interp

# functions_math
* Implemented most of the vectorized math functions for tensor type with Tensor\[T\] outputs, but many functions are still left as mentioned in the comments.
* Implement a _math_ function for the SIMD\[bool\] type outputs

