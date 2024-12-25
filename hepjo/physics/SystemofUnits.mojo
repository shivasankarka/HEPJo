"""
# ----------------------------------------------------------------------
# HEP coherent system of Units
#
# This file has been provided to CLHEP by Geant4 (simulation toolkit for HEP).
#
# The basic units are:
# - millimeter              (mm)
# - nanosecond              (ns)
# - Mega electron Volt      (MeV)
# - positron charge         (eplus)
# - degree Kelvin           (K)
# - the amount of substance (mole)
# - luminous intensity      (candela)
# - radian                  (rad)
# - steradian               (sr)
#
# Below is a non-exhaustive list of derived and practical units
# (mostly SI units). You can add your own units.
#
# The SI numerical value of the positron charge is defined here,
# as it is needed for conversion factor: positron charge = e_SI (coulomb)
#
# The other physical constants are defined in the header file:
# PhysicalConstants.h
#
# Authors: M. Maire, S. Giani
#
# History:
#
# 24.08.17   Created.
"""

alias     pi  :Float64 =  3.14159265358979323846;
alias  twopi  :Float64 =  2*pi;
alias halfpi  :Float64 =  pi/2;
alias     pi2 :Float64 =  pi*pi;
 
#  Length [L]

alias millimeter  :Float64 =  1.;                        
alias millimeter2 :Float64 =  millimeter*millimeter;
alias millimeter3 :Float64 =  millimeter*millimeter*millimeter;

alias centimeter  :Float64 =  10.*millimeter;   
alias centimeter2 :Float64 =  centimeter*centimeter;
alias centimeter3 :Float64 =  centimeter*centimeter*centimeter;

alias meter  :Float64 =  1000.*millimeter;                  
alias meter2 :Float64 =  meter*meter;
alias meter3 :Float64 =  meter*meter*meter;

alias kilometer :Float64 =  1000.*meter;                   
alias kilometer2 :Float64 =  kilometer*kilometer;
alias kilometer3 :Float64 =  kilometer*kilometer*kilometer;

alias parsec :Float64 =  3.0856775807e+16*meter;

alias micrometer :Float64 =  1.e-6 *meter;             
alias  nanometer :Float64 =  1.e-9 *meter;
alias  angstrom  :Float64 =  1.e-10*meter;
alias  fermi     :Float64 =  1.e-15*meter;

alias      barn :Float64 =  1.e-28*meter2;
alias millibarn :Float64 =  1.e-3 *barn;
alias microbarn :Float64 =  1.e-6 *barn;
alias  nanobarn :Float64 =  1.e-9 *barn;
alias  picobarn :Float64 =  1.e-12*barn;

#  symbols
alias nm  :Float64 =  nanometer;                        
alias um  :Float64 =  micrometer;                        

alias mm  :Float64 =  millimeter;                        
alias mm2 :Float64 =  millimeter2;
alias mm3 :Float64 =  millimeter3;

alias cm  :Float64 =  centimeter;   
alias cm2 :Float64 =  centimeter2;
alias cm3 :Float64 =  centimeter3;

alias liter :Float64 =  1.e+3*cm3;
alias  L :Float64 =  liter;
alias dL :Float64 =  1.e-1*liter;
alias cL :Float64 =  1.e-2*liter;
alias mL :Float64 =  1.e-3*liter;       

alias m  :Float64 =  meter;                  
alias m2 :Float64 =  meter2;
alias m3 :Float64 =  meter3;

alias km  :Float64 =  kilometer;                   
alias km2 :Float64 =  kilometer2;
alias km3 :Float64 =  kilometer3;

alias pc :Float64 =  parsec;


#  Angle

alias radian      :Float64 =  1.;                  
alias milliradian :Float64 =  1.e-3*radian;
alias degree :Float64 =  (pi/180.0)*radian;

alias   steradian :Float64 =  1.;

#  symbols
alias rad  :Float64 =  radian;
alias mrad :Float64 =  milliradian;
alias sr   :Float64 =  steradian;
alias deg  :Float64 =  degree;


#  Time [T]

alias nanosecond  :Float64 =  1.;
alias second      :Float64 =  1.e+9 *nanosecond;
alias millisecond :Float64 =  1.e-3 *second;
alias microsecond :Float64 =  1.e-6 *second;
alias picosecond  :Float64 =  1.e-12*second;

alias minute :Float64 =  60*second;
alias hour   :Float64 =  60*minute;
alias day    :Float64 =  24*hour;
alias year   :Float64 =  365*day;  

alias hertz :Float64 =  1./second;
alias kilohertz :Float64 =  1.e+3*hertz;
alias megahertz :Float64 =  1.e+6*hertz;

#  symbols
alias ns :Float64 =  nanosecond;
alias  s :Float64 =  second;
alias ms :Float64 =  millisecond;
alias us :Float64 =  microsecond;
alias ps :Float64 =  picosecond;


#  Electric charge [Q]

alias eplus :Float64 =  1. ; positron charge
alias e_SI  :Float64 =  1.602176634e-19; positron charge in coulomb
alias coulomb :Float64 =  eplus/e_SI; coulomb :Float64 =  6.24150 e+18 * eplus


#  Energy [E]

alias megaelectronvolt :Float64 =  1. ;
alias     electronvolt :Float64 =  1.e-6*megaelectronvolt;
alias kiloelectronvolt :Float64 =  1.e-3*megaelectronvolt;
alias gigaelectronvolt :Float64 =  1.e+3*megaelectronvolt;
alias teraelectronvolt :Float64 =  1.e+6*megaelectronvolt;
alias petaelectronvolt :Float64 =  1.e+9*megaelectronvolt;
alias millielectronvolt :Float64 =  1.e-9*megaelectronvolt;  

alias joule :Float64 =  electronvolt/e_SI; joule :Float64 =  6.24150 e+12 * MeV

#  symbols
alias MeV :Float64 =  megaelectronvolt;
alias  eV :Float64 =  electronvolt;
alias keV :Float64 =  kiloelectronvolt;
alias GeV :Float64 =  gigaelectronvolt;
alias TeV :Float64 =  teraelectronvolt;
alias PeV :Float64 =  petaelectronvolt;


#  Mass [E][T^2][L^-2]

alias  kilogram :Float64 =  joule*second*second/(meter*meter);   
alias      gram :Float64 =  1.e-3*kilogram;
alias milligram :Float64 =  1.e-3*gram;

 symbols
alias  kg :Float64 =  kilogram;
alias   g :Float64 =  gram;
alias  mg :Float64 =  milligram;


#  Power [E][T^-1]

alias watt :Float64 =  joule/second; watt :Float64 =  6.24150 e+3 * MeV/ns


#  Force [E][L^-1]

alias newton :Float64 =  joule/meter; newton :Float64 =  6.24150 e+9 * MeV/mm


#  Pressure [E][L^-3]

#define pascal hep_pascal                           a trick to avoid warnings 
alias hep_pascal :Float64 =  newton/m2;    pascal :Float64 =  6.24150 e+3 * MeV/mm3
alias bar        :Float64 =  100000*pascal;  bar    :Float64 =  6.24150 e+8 * MeV/mm3
alias atmosphere :Float64 =  101325*pascal;  atm    :Float64 =  6.32420 e+8 * MeV/mm3


#  Electric current [Q][T^-1]

alias      ampere :Float64 =  coulomb/second;  ampere :Float64 =  6.24150 e+9 * eplus/ns
alias milliampere :Float64 =  1.e-3*ampere;
alias microampere :Float64 =  1.e-6*ampere;
alias  nanoampere :Float64 =  1.e-9*ampere;


#  Electric potential [E][Q^-1]

alias megavolt :Float64 =  megaelectronvolt/eplus;
alias kilovolt :Float64 =  1.e-3*megavolt;
alias     volt :Float64 =  1.e-6*megavolt;


#  Electric resistance [E][T][Q^-2]

alias ohm :Float64 =  volt/ampere; ohm :Float64 =  1.60217e-16*(MeV/eplus)/(eplus/ns)


#  Electric capacitance [Q^2][E^-1]

alias farad :Float64 =  coulomb/volt; farad :Float64 =  6.24150e+24 * eplus/Megavolt
alias millifarad :Float64 =  1.e-3*farad;
alias microfarad :Float64 =  1.e-6*farad;
alias  nanofarad :Float64 =  1.e-9*farad;
alias  picofarad :Float64 =  1.e-12*farad;


#  Magnetic Flux [T][E][Q^-1]

alias weber :Float64 =  volt*second; weber :Float64 =  1000*megavolt*ns


#  Magnetic Field [T][E][Q^-1][L^-2]

alias tesla     :Float64 =  volt*second/meter2; tesla :Float64 = 0.001*megavolt*ns/mm2

alias gauss     :Float64 =  1.e-4*tesla;
alias kilogauss :Float64 =  1.e-1*tesla;


#  Inductance [T^2][E][Q^-2]

alias henry :Float64 =  weber/ampere; henry :Float64 =  1.60217e-7*MeV*(ns/eplus)**2


 Temperature

alias kelvin :Float64 =  1.;


 Amount of substance

alias mole :Float64 =  1.;


#  Activity [T^-1]

alias becquerel :Float64 =  1./second ;
alias curie :Float64 =  3.7e+10 * becquerel;
alias kilobecquerel :Float64 =  1.e+3*becquerel;
alias megabecquerel :Float64 =  1.e+6*becquerel;
alias gigabecquerel :Float64 =  1.e+9*becquerel;
alias millicurie :Float64 =  1.e-3*curie;
alias microcurie :Float64 =  1.e-6*curie;
alias Bq :Float64 =  becquerel;
alias kBq :Float64 =  kilobecquerel;
alias MBq :Float64 =  megabecquerel;
alias GBq :Float64 =  gigabecquerel;
alias Ci :Float64 =  curie;
alias mCi :Float64 =  millicurie;
alias uCi :Float64 =  microcurie;


#  Absorbed dose [L^2][T^-2]

alias      gray :Float64 =  joule/kilogram ;
alias  kilogray :Float64 =  1.e+3*gray;
alias milligray :Float64 =  1.e-3*gray;
alias microgray :Float64 =  1.e-6*gray;


#  Luminous intensity [I]

alias candela :Float64 =  1.;


#  Luminous flux [I]

alias lumen :Float64 =  candela*steradian;


#  Illuminance [I][L^-2]

alias lux :Float64 =  lumen/meter2;


#  Miscellaneous

alias perCent     :Float64 =  0.01 ;
alias perThousand :Float64 =  0.001;
alias perMillion  :Float64 =  0.000001;