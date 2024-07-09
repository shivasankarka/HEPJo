from hepjo import *

fn main() raises:
    var rhn_attr = ParticleAttributes[DType.float32](name = "RHN", mass = 1.0, width = 1.0, charge = 0.0, spin = 1, PID = 32, type="lepton")
    # var rhn = Particle[DType.float32](attr = rhn_attr, momentum = LorentzVector[DType.float32](1, 2, 3, 4), pos = Vector3D[DType.float32](1, 2, 3))
    var rhn = Particle[DType.float32](attr = rhn_attr)
    print(rhn)

    var gun = ParticleGun[DType.float32](particle = rhn)
    print(gun)

    for i in range(10):
        print("initial position: ", gun.particle.pos)
        gun.generate(energy = 2.0, seed_value = i)
        gun.simulate(t = 1.0)
        print("final position: ", gun.particle.pos)

    # var momenta_list = gun.generate(energy = 2.0, N = 10)
    # print(momenta_list[0])



