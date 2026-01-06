import reentripy as rpy
import numpy as np
from pystdatm import density


# ------------------------------
# Starship parameters (empty)
# ------------------------------
cl = 1.2
cd = 1.3
area = 63.82       # m^2
mass = 120_000.0    # kg

# Peak heating: IFT 11 T+ 51' 51'' --> 24783 kmh 70.2 km

peak_heating_speed = 24783/3.6
peak_heating_alt = 70200.0

peak_heating_rho = density(peak_heating_alt)

nose_radius = 3

k = 1.7415e-4  # (Earth)
qc_max = k * np.sqrt(peak_heating_rho/nose_radius) * peak_heating_speed ** 3



# Create spacecraft
boca_chica_lat = 25.9972   # degrees North
boca_chica_lon = -97.1566 # degrees East (West is negative)
sc = rpy.Spacecraft(cl=cl, cd=cd, A=area, m=mass, max_qc=qc_max, nose_radius=nose_radius, landing_lat=boca_chica_lat, landing_lon=boca_chica_lon)

sc.load_aero_tables(
    "Starship Aero Data/wpd_starship_cl.csv",
    "Starship Aero Data/wpd_starship_cd.csv"
)

# ------------------------------
# Orbit definition: Conditions for IFT test flights ( more or less )
# ------------------------------
apogee = 213_000.0
perigee = -15_000.0
altitude = 200_000.0   # current altitude

inclination = np.deg2rad(26.8)     # Starship-like
arg_perigee = np.deg2rad(148.5)
raan = np.deg2rad(165.0)



# ------------------------------
# Generate Cartesian state
# ------------------------------
sc.keplerian_initial_conditions(
    apogee=apogee,
    perigee=perigee,
    altitude=altitude,
    inclination=inclination,
    arg_perigee=arg_perigee,
    raan=raan,
    true_anomaly_sign=-1  # descending branch (reentry)
)



sc.banking_angle = 0
sc.alpha = 50

sc.banking_angle = 0
sc.alpha = 50

times, altitudes, speeds, machs, bank_angles, g_forces, descent_rates, positions, lon, lat, heat_loads, heat_fluxes, aoas, sogs_vecs = sc.run_reentry(gif=False, controller="aPQC", DTLH=True) # With controller adjusting bank trying to keep max heating (DOESN'T REALLY WORK --> HIGH Gs)


