from tensor import Tensor, TensorShape
from builtin.dtype import DType
from collections.vector import InlinedFixedVector

from testing import assert_equal
from scijo import Vector2D, Vector3D, LorentzVector
import scijo as sj
import math

fn main() raises:
    # * Vector3D
    ## Initialization (Default type: DType.float64)
    var v1 = Vector3D() # Initializes a empty 3D vector with type=DType.float64  v1 = [0.0, 0.0, 0.0]
    var v2 = Vector3D(1.0, 2.0, 3.0) # Initializes a 3D vector with type=DType.float64 v2 = [1.0, 2.0, 3.0]
    var v2 = Vector3D[DType.float64].frompoint(x=0.0, y=3.0, z=4.0) # Initializes a 3D vector with type=DType.float64 from given points x,y,z, v2 = [0.0, 3.0, 4.0]
    var v3 = Vector3D.fromvector(v2) # Initializes a 3D vector using another 3D vector as argument, v3 =v2
    var v_sph = Vector3D.fromsphericalcoords(r=1.0, theta=0.5, phi=1.0) # Initializes a 3D vector with given spherical coordinates
    var v_cyl = Vector3D.fromcylindricalcoodinates(rho=1.0, phi=1.0, z=3.0) # Initializes a 3D vector with given cylindrical coordinates
    var v_fromlist = Vector3D.fromlist(List[DType.float64](3.0,1.0,2.0)) # Initializes a 3D vector with given elements of List

    ## Basic arithmetic (Vectorized fucntions hence zoom zoom fast)
    var v_sum_vector = v1 + v2 # adds two 3D vectors 
    var v_sum_scalar = v1 + 2.0 # adds a 3D vector and scalar of same type
    # similar setup for subtraction, multiplication, division

    # Extra operations
    var v_elementwise_mul = v1*v2 # multiplies two 3D vectors elementwise and stores the resultant 3D vector in v3
    var v_dot = v1@v2 # this performs Dot product of two 3D vectors and store the resultant scalar of same type in v4
    var v_dot_alt = v1.dot(v2) # does the same dot product of of these two 3D vectors, just extra syntactic sugar.
    var v_cross = v1.cross(v2) # does cross product of v1, v2 and returns the resultant vector
    var v_act_function = v1.act[math.exp]() # Takes the v1 vector and performs f(v1) elementwise where f is some function which is SIMD compatible (math library)
    var v_pow = v1**2 # performse elementwise pow(v1) and returns the resultant 3D vector.

    # getter, setters
    var v_x = v2.x() # returns the x coordinate, similar syntax for y, z coodinates
    v2[0] = 10.0 # changes the x element to 10.0, similar syntax for y, z coodinates
    v2.x(10.0) # same as above, changes the x element to given value, similar syntax for y, z coodinates
    v1.set(1.0, 3.0, 5.0) # sets all elements at once
    var v_list = v1.tolist() # returns a list with the vector elements  
    var v_copy = v1.copy() # returns a copy of the vector

    # properties
    var v_rho = v2.rho() # returns the rho value sqrt(x**2 + y**2)
    var v_mag = v2.mag() # returns the magnitude of the vector
    var v_mag = v2.mag2() # returns the magnitude squared of the vector
    var v_r = v2.r() # same as mag()
    var v_costheta = v1.costheta() # returns the cosine of the angle between v1 vector and z axis
    var v_theta = v1.theta() # returns the theta angle between v1 vector and z axis
    var v_phi = v1.phi() # returns the phi angle in x-y plane from x axis
    var v_unit = v1.unit() # returns the corresponding unit vector
    if v1: # can be used to check if vector is zero or non zero
        print("vector is non zero")
    
    # rotations
    var v1_rot_v2 = v1.rotate(v2, 10) # rotates the v1 vector with v2 vector as the axis (first argument) with the given angle (second argument)
    var v1_rotx = v1.rotate_x() # rotates the v1 vector about the x axis with the given angle, similarly for y and z
    var v1_cosangle = v1.cos_angle(v2) # calculates the cosine of angle between v1 and v2 vector
    var v1_angle = v1.angle(v2) # calculates the angle between v1 and v2 vector
    print(v1.isparallel(v2)) # checks if v1 and v2 is parallel and returns Boolean (True or False)
    print(v1.isantiparallel(v2)) # checks if v1 and v2 is anti parallel and returns Boolean (True or False)
    print(v1.isperpendicular(v2)) # checks if v1 and v2 is perpendicular and returns Boolean (True or False)






    









