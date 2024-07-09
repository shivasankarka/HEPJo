from utils.numerics import inf
from random import seed, random_float64, random_si64
from math import sqrt

from .lorentzvector import LorentzVector
from .vector import Vector3D
from .constants import pi


struct ParticleAttributes[dtype: DType = DType.float64]():
    var name: String
    var mass: SIMD[dtype, 1]
    var width: SIMD[dtype, 1]
    var charge: SIMD[dtype, 1]
    var spin: Int
    var parity: Int
    var isospin: Int
    var isospin3: Int
    var type: String
    var lepton_number: Int
    var baryon_number: Int
    var PID: Int
    var stable: Bool
    var lifetime: SIMD[dtype, 1]

    fn __init__(
        inout self,
        name: String = "",
        mass: SIMD[dtype, 1] = 0.0,
        width: SIMD[dtype, 1] = 0.0,
        charge: SIMD[dtype, 1] = 0.0,
        spin: Int = 0,
        parity: Int = 0,
        isospin: Int = 0,
        isospin3: Int = 0,
        type: String = "particle",
        lepton_number: Int = 0,
        baryon_number: Int = 0,
        PID: Int = 0,
        stable: Bool = True,
        lifetime: SIMD[dtype, 1] = inf[dtype](),
    ):
        self.name = name
        self.mass = mass
        self.width = width
        self.charge = charge
        self.spin = spin
        self.parity = parity
        self.isospin = isospin
        self.isospin3 = isospin3
        self.type = type
        self.lepton_number = lepton_number
        self.baryon_number = baryon_number
        self.PID = PID
        self.stable = stable
        self.lifetime = lifetime
        print(name + " is created: mass = " + str(mass) + "\n")

    @always_inline("nodebug")
    fn __copyinit__(inout self, other: Self):
        self.name = other.name
        self.mass = other.mass
        self.width = other.width
        self.charge = other.charge
        self.spin = other.spin
        self.parity = other.parity
        self.isospin = other.isospin
        self.isospin3 = other.isospin3
        self.type = other.type
        self.lepton_number = other.lepton_number
        self.baryon_number = other.baryon_number
        self.PID = other.PID
        self.stable = other.stable
        self.lifetime = other.lifetime

    @always_inline("nodebug")
    fn __moveinit__(inout self, owned existing: Self):
        self.name = existing.name
        self.mass = existing.mass
        self.width = existing.width
        self.charge = existing.charge
        self.spin = existing.spin
        self.parity = existing.parity
        self.isospin = existing.isospin
        self.isospin3 = existing.isospin3
        self.type = existing.type
        self.lepton_number = existing.lepton_number
        self.baryon_number = existing.baryon_number
        self.PID = existing.PID
        self.stable = existing.stable
        self.lifetime = existing.lifetime

    @always_inline("nodebug")
    fn __str__(self) -> String:
        return (
            "******************"
            + "\n"
            + "Particle: "
            + self.name
            + "\n"
            + "\t"
            + "Mass: "
            + str(self.mass)
            + "\t"
            + "Width: "
            + str(self.width)
            + "\t"
            + "Charge: "
            + str(self.charge)
            + "\n"
            + "\t"
            + "Spin: "
            + str(self.spin)
            + "\t\t"
            + "Parity: "
            + str(self.parity)
            + "\t"
            + "Isospin: "
            + str(self.isospin)
            + "\t"
            + "Isospin3: "
            + str(self.isospin3)
            + "\n"
            + "\t"
            + "Type: "
            + str(self.type)
            + "\t"
            + "Lepton Number: "
            + str(self.lepton_number)
            + "\t"
            + "Baryon Number: "
            + str(self.baryon_number)
            + "\n"
            + "\t"
            + "PID: "
            + str(self.PID)
            + "\t\t"
            + "Stable: "
            + str(self.stable)
            + "\t"
            + "Lifetime: "
            + str(self.lifetime)
            + "\n"
            + "******************"
            + "\n"
        )


struct Particle[
    dtype: DType = DType.float64,
](Stringable):
    var attr: ParticleAttributes[dtype]
    var momentum: LorentzVector[dtype]
    var pos: Vector3D[dtype]

    fn __init__(
        inout self,
        attr: ParticleAttributes[dtype]
    ):
        self.attr = attr
        self.momentum = LorentzVector[dtype](0.0, 0.0, 0.0, 0.0)
        self.pos = Vector3D[dtype](0.0, 0.0, 0.0)

    fn __init__(
        inout self, 
        attr: ParticleAttributes[dtype],
        momentum: LorentzVector[dtype],
        pos: Vector3D[dtype]
    ):
        self.attr = attr
        self.momentum = momentum
        self.pos = pos

    fn __copyinit__(inout self, other: Self):
        self.attr = other.attr
        self.momentum = other.momentum
        self.pos = other.pos

    fn __moveinit__(inout self, owned existing: Self):
        self.attr = existing.attr
        self.momentum = existing.momentum
        self.pos = existing.pos

    fn __str__(self) -> String:
        return (
            "*****************"
            + "\n"
            + "Name: "
            + self.attr.name
            + "\n"
            + "Momentum: "
            + str(self.momentum)
            + "\n"
            + "Position: "
            + str(self.pos)
            + "\n"
            + "*****************"
            + "\n"
        )


struct ParticleGun[dtype: DType = DType.float64]():
    var seed: Int
    var x: SIMD[dtype, 1]
    var y: SIMD[dtype, 1]
    var z: SIMD[dtype, 1]
    var particle: Particle[dtype]

    fn __init__(
        inout self,
        particle: Particle[dtype],
        seed: Int = 42,
        x: SIMD[dtype, 1] = 0.0,
        y: SIMD[dtype, 1] = 0.0,
        z: SIMD[dtype, 1] = 0.0,
    ):
        self.particle = particle
        self.seed = seed
        self.x = x
        self.y = y
        self.z = z

    fn generate(
        inout self, energy: SIMD[dtype, 1], seed_value: Int = 42
    ) raises:
        if energy < self.particle.attr.mass:
            raise Error(
                "[ParticleGun -> generate] \n Invalid energy: Energy cannot be"
                " less than rest mass."
            )
        self.seed = seed_value
        seed(self.seed)
        var p = sqrt(energy**2 - self.particle.attr.mass**2)
        var theta = random_float64(0.0, pi)
        var phi = random_float64(0.0, 2 * pi)
        var pxpypz = Vector3D[dtype].fromsphericalcoords(
            r=p, theta=theta, phi=phi
        )
        var pxpypzm = LorentzVector[dtype](vector3d=pxpypz, t=energy)
        self.particle.momentum = pxpypzm

    # TODO : Figure out why we can malloc error while using this
    fn generate(
        self, energy: SIMD[dtype, 1], N: Int
    ) raises -> List[LorentzVector[dtype]]:
        if energy < self.particle.attr.mass:
            raise Error(
                "[ParticleGun -> generate] Invalid energy: Energy cannot be"
                " less than rest mass."
            )
        var momenta_list: List[LorentzVector[dtype]] = List[
            LorentzVector[dtype]
        ]()
        for _ in range(N):
            seed(self.seed)
            var p = sqrt(energy**2 - self.particle.attr.mass**2)
            var theta = random_float64(0.0, pi)
            var phi = random_float64(0.0, 2 * pi)
            var pxpypz = Vector3D[dtype].fromsphericalcoords(
                r=p, theta=theta, phi=phi
            )
            var pxpypzm = LorentzVector[dtype](vector3d=pxpypz, t=energy)
            momenta_list.append(pxpypzm)
        return momenta_list

    fn __copyinit__(inout self, other: Self):
        self.seed = other.seed
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.particle = other.particle

    fn __moveinit__(inout self, owned existing: Self):
        self.seed = existing.seed
        self.x = existing.x
        self.y = existing.y
        self.z = existing.z
        self.particle = existing.particle

    fn __str__(self) -> String:
        return (
            "******************"
            + "\n"
            + "Particle Rifle"
            + "\n"
            + "Generates: "
            + str(self.particle.attr.name)
            + "\n"
            + "[Position: "
            + str(self.x)
            + ", "
            + str(self.y)
            + ", "
            + str(self.z)
            + "]"
            + "\n"
            + "******************"
            + "\n"
        )

    fn simulate(inout self, t: SIMD[dtype, 1], verbose: Bool = False):
        if verbose:
            print("Simulation started for " + self.particle.attr.name + "\n")
            print("Time: " + str(t) + "\n")
        var vel = self.particle.momentum.vector()
        self.particle.pos = self.particle.pos +  vel * (t / self.particle.momentum.e())  # calculate new position
        if verbose:
            print("New position: " + str(self.particle.pos) + "\n")
            print("Simulation ended for " + self.particle.attr.name + "\n")


