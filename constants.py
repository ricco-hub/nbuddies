from pint import UnitRegistry


# Store unit defintions, relationships
# and handle unit conversions
ureg = UnitRegistry()
# Define solar mass in kilograms
ureg.define("M_sol = 1.98847E30 * kg")

# Constants
# https://physics.nist.gov/cgi-bin/cuu/Value?bg
G = 6.67e-11 * (
    ureg.m**3 / ureg.kg / ureg.s**2
)  # Newton's constant [m^3 / kg / s^2]
# Convert to new units [km^2 * kpc / (M_sol * s^2)]
G = G.to((ureg.km**2 * ureg.kpc) / (ureg.M_sol * ureg.s**2))

# Conversions
kpc_to_km = (1 * ureg.kpc).to(ureg.km)  # km per kpc
