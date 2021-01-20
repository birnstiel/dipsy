"natural constants in CGS"
import astropy.constants as c
import astropy.units as u

h = c.h.cgs.value
c_light = c.c.cgs.value
k_B = c.k_B.cgs.value
pc = c.pc.cgs.value
jy_sas = (1 * u.Jy / u.arcsec**2).cgs.value
year = (1 * u.year).cgs.value
au = c.au.cgs.value
R_sun = c.R_sun.cgs.value
M_sun = c.M_sun.cgs.value
Grav = c.G.cgs.value
sigma_sb = c.sigma_sb.cgs.value

mu = 2.3
