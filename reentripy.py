import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from pystdatm import density, temperature, speed_of_sound
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import brentq

import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from math import acos, sin, cos

def short_great_circle_distance(lat1, lon1, lat2, lon2, radius=6371000.0):
    # Compute central angle
    cos_sigma = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon2 - lon1)
    # Clamp to [-1, 1] to avoid rounding errors
    cos_sigma = max(-1.0, min(1.0, cos_sigma))
    sigma = acos(cos_sigma)
    # Shortest distance along sphere
    if sigma > np.pi:
        sigma = 2*np.pi - sigma
    return sigma * radius


def eci_to_lonlat(r_vec_eci, t, planet_radius=6_371_000.0):
    """
    Converts ECI Cartesian position vector(s) to geodetic latitude and longitude (radians),
    accounting for planetary rotation at time t (s since epoch).

    Parameters
    ----------
    r_vec_eci : ndarray
        Nx3 array of ECI positions (m) or shape (3,)
    t : float or ndarray
        Time(s) since reference epoch (s)
    planet_radius : float
        Radius of planet (m)

    Returns
    -------
    lon, lat : ndarray
        Longitude and latitude in radians (length N)
    """
    r = np.atleast_2d(r_vec_eci).astype(float)

    # --- EARLY EXIT FOR EMPTY TRAJECTORIES ---
    if r.shape[0] == 0:
        return np.array([]), np.array([])
    N = r.shape[0]

    # Earth's rotation rate (rad/s)
    omega = 2.0 * np.pi / 86164.0

    # Time handling
    t = np.asarray(t, dtype=float)

    if t.ndim == 0:
        theta = omega * np.full(N, t)
    elif t.ndim == 1 and t.size == N:
        theta = omega * t
    else:
        raise ValueError(
            f"Time array must be scalar or length {N}, got shape {t.shape}"
        )

    # Rotation: ECI → ECEF (about Z)
    c = np.cos(-theta)
    s = np.sin(-theta)

    x = c * r[:, 0] - s * r[:, 1]
    y = s * r[:, 0] + c * r[:, 1]
    z = r[:, 2]

    # Longitude / latitude
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.sqrt(x*x + y*y + z*z))

    return lon, lat



def altitude_for_density(rho_target, h_min=0, h_max=100_000):
    """
    Find altitude (m) where atmospheric density equals rho_target (kg/m^3)
    using pystdatm.
    h_min, h_max : search range in meters
    """
    # Define function: f(h) = density(h) - rho_target
    f = lambda h: atmospheric_properties(h)[0] - rho_target

    try:
        h_sol = brentq(f, h_min, h_max)
    except ValueError:
        # If solution not found in range, return None
        h_sol = None
    return h_sol


def atmospheric_properties(altitude_m):
    """
    Altitude-only hybrid atmosphere:
    - US Standard Atmosphere below 79 km
    - Exponential thermosphere above 79 km
    """

    gamma = 1.4
    R = 287.05  # J/(kg·K)

    # ----------------------------
    # Lower atmosphere (USSA)
    # ----------------------------
    if altitude_m <= 79_000.0:
        rho = density(altitude_m)
        temp = temperature(altitude_m)
        sound = np.sqrt(gamma * R * temp)

        if not np.isfinite(rho) or rho < 0:
            rho = 0.0
        if not np.isfinite(temp) or temp <= 0:
            temp = 200.0

        return rho, temp, sound

    # ----------------------------
    # Upper atmosphere (Exponential)
    # ----------------------------
    h_transition = 79_000.0

    # Anchor slightly BELOW transition for numerical stability
    h_anchor = 79_000.0
    rho0 = density(h_anchor)
    T0 = temperature(h_anchor)

    g0 = 9.80665  # m/s^2

    # Thermospheric temperature profile (used for density shaping only)
    T_inf = 900.0  # K
    temp = T_inf - (T_inf - T0) * np.exp(-(altitude_m - h_anchor) / 40_000.0)

    # Local scale height from temperature
    H = R * temp  / g0

    rho = rho0 * np.exp(-(altitude_m - h_anchor) / H)



    # Freeze speed of sound above transition
    sound = np.sqrt(gamma * R * T0)

    return rho, temp, sound
#
# test_altitudes = np.arange(0, 140_000.0, 10)
#
# # Compute density (vectorized safely)
# test_rhos = np.array([
#     atmospheric_properties(h)[0] for h in test_altitudes
# ])
#
# # Plot
# plt.figure(figsize=(7, 5))
# plt.semilogy(test_altitudes / 1000, test_rhos, color="blue")
#
# plt.xlabel("Altitude (km)")
# plt.ylabel("Density (kg/m³)")
# plt.title("Atmospheric Density vs Altitude")
# plt.grid(True, which="both", alpha=0.3)
#
# plt.tight_layout()
# plt.show()

class Spacecraft:
    def __init__(self, cl, cd, A, m, max_qc=None, nose_radius=3, alpha=50, landing_lat=0, landing_lon=0):
        self.cl = cl
        self.cd = cd
        self.Area = A
        self.mass = m
        self.banking_angle = 0
        self.alpha = alpha
        self.nose_R = nose_radius
        self.max_qc = max_qc
        self.landing_lat = landing_lat
        self.landing_lon = landing_lon

    def load_aero_tables(self, cl_csv, cd_csv):

        def parse(path):
            df = pd.read_csv(path)

            M, AOA, C = [], [], []

            cols = df.columns
            i = 0
            while i < len(cols):
                if "Ma:" in cols[i]:
                    mach = float(cols[i].split(":")[1])

                    aoa = df.iloc[2:, i].astype(float).values
                    coeff = df.iloc[2:, i + 1].astype(float).values

                    mask = np.isfinite(aoa) & np.isfinite(coeff)

                    M.extend([mach] * np.sum(mask))
                    AOA.extend(aoa[mask])
                    C.extend(coeff[mask])

                    i += 2
                else:
                    i += 1

            return np.array(M), np.array(AOA), np.array(C)

        # Parse data
        M_cl, AOA_cl, CL = parse(cl_csv)
        M_cd, AOA_cd, CD = parse(cd_csv)

        # Store raw data (THIS WAS MISSING)
        self._mach_data = M_cl
        self._aoa_data = AOA_cl
        self._cl_data = CL
        self._cd_data = CD

        # Build interpolators
        self._cl_interp = LinearNDInterpolator(
            np.column_stack((M_cl, AOA_cl)), CL
        )
        self._cd_interp = LinearNDInterpolator(
            np.column_stack((M_cd, AOA_cd)), CD
        )

        self.aero_tables_loaded = True

        machs = self._mach_data
        aoas = self._aoa_data

        self.mach_min = machs.min() + 0.1
        self.mach_max = machs.max() - 0.1
        self.aoa_min = aoas.min() + 0.1
        self.aoa_max = aoas.max() - 0.1

    def get_cl_cd(self, mach, aoa_deg):
        if aoa_deg is None:
            raise ValueError("AOA is None — controller failed to set a valid alpha")

        if self.aoa_max is None or self.aoa_min is None:
            raise ValueError("AOA limits not initialized")

        if not np.isfinite(mach):
            mach = self.mach_max
        elif mach > self.mach_max:
            mach = self.mach_max
        elif mach < self.mach_min:
            mach = self.mach_min

        if aoa_deg > self.aoa_max:
            aoa_deg = self.aoa_max
        elif aoa_deg < self.aoa_min:
            aoa_deg = self.aoa_min

        CL = self._cl_interp(mach, aoa_deg)
        CD = self._cd_interp(mach, aoa_deg)

        CL = np.where(np.isfinite(CL), CL, np.nan)
        CD = np.where(np.isfinite(CD), CD, np.nan)
        return float(CL), float(CD)

    def plot_aero_interpolation(
            self,
            mach_min=None,
            mach_max=None,
            aoa_min=None,
            aoa_max=None,
            n_mach=60,
            n_aoa=60
    ):
        """
        Plot 2D Mach–AOA interpolation surfaces for:
          - CL
          - CD
          - CL/CD

        Uses scattered 2-D interpolators.
        """

        # ----------------------------
        # Safety check
        # ----------------------------
        for attr in ("_cl_interp", "_cd_interp", "_mach_data", "_aoa_data"):
            if not hasattr(self, attr):
                raise RuntimeError(
                    "Call load_aero_tables_scattered() first."
                )

        # ----------------------------
        # Bounds
        # ----------------------------
        # machs = self._mach_data
        # aoas = self._aoa_data

        # self.mach_min = mach_min if mach_min is not None else self.mach_min
        # self.mach_max = mach_max if mach_max is not None else self.mach_max
        # self.aoa_min = aoa_min if aoa_min is not None else self.aoa_min
        # self.aoa_max = aoa_max if aoa_max is not None else self.aoa_max

        # ----------------------------
        # Surface grid
        # ----------------------------
        mach_grid = np.linspace(mach_min, mach_max, n_mach)
        aoa_grid = np.linspace(aoa_min, aoa_max, n_aoa)

        M, AOA = np.meshgrid(mach_grid, aoa_grid)

        CL = self._cl_interp(M, AOA)
        CD = self._cd_interp(M, AOA)

        CL = np.where(np.isfinite(CL), CL, np.nan)
        CD = np.where(np.isfinite(CD), CD, np.nan)
        LDR = np.where(CD > 0, CL / CD, np.nan)

        # ----------------------------
        # Scatter points (COMMON SET)
        # ----------------------------
        mach_sc = self._mach_data
        aoa_sc = self._aoa_data

        cl_sc = self._cl_interp(mach_sc, aoa_sc)
        cd_sc = self._cd_interp(mach_sc, aoa_sc)

        valid = np.isfinite(cl_sc) & np.isfinite(cd_sc) & (cd_sc > 0)

        mach_sc = mach_sc[valid]
        aoa_sc = aoa_sc[valid]
        cl_sc = cl_sc[valid]
        cd_sc = cd_sc[valid]
        ldr_sc = cl_sc / cd_sc

        # ----------------------------
        # Plot
        # ----------------------------
        fig = plt.figure(figsize=(18, 7))

        plots = [
            ("CL", CL, cl_sc, "viridis"),
            ("CD", CD, cd_sc, "plasma"),
            ("CL/CD", LDR, ldr_sc, "inferno"),
        ]

        for i, (label, Z, scatter_vals, cmap) in enumerate(plots):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")

            surf = ax.plot_surface(
                M, AOA, Z,
                cmap=cmap,
                linewidth=0,
                alpha=0.9
            )

            ax.scatter(
                mach_sc,
                aoa_sc,
                scatter_vals,
                color="k",
                s=14,
                alpha=0.75
            )

            ax.set_xlabel("Mach")
            ax.set_ylabel("AOA (deg)")
            ax.set_zlabel(label)
            ax.set_title(f"{label} Mach–AOA Interpolation")

            fig.colorbar(surf, ax=ax, shrink=0.6)

        plt.tight_layout()
        plt.show()

    def initial_conditions(
        self,
        altitude,
        longitude,
        latitude,
        radial_velocity,
        tangential_velocity,
        inclination,
        planet_radius=6_371_000.0
    ):
        """
        altitude             : m above planet radius
        longitude, latitude  : radians
        inclination           : radians (0 = east, pi/2 = north)
        radial_velocity       : m/s (positive outward)
        tangential_velocity   : m/s (local horizontal magnitude)
        planet_radius         : m
        """

        # Store scalars
        self.altitude = altitude
        self.longitude = longitude
        self.latitude = latitude

        lat = latitude
        lon = longitude
        i = inclination

        # --- Local unit vectors (ECEF) ---

        # Radial (up)
        e_r = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat)
        ])

        # East
        e_east = np.array([
            -np.sin(lon),
             np.cos(lon),
             0.0
        ])

        # North
        e_north = np.array([
            -np.sin(lat) * np.cos(lon),
            -np.sin(lat) * np.sin(lon),
             np.cos(lat)
        ])

        # Tangential direction in local horizontal plane
        e_t = np.cos(i) * e_east + np.sin(i) * e_north

        # --- Cartesian position ---
        r = planet_radius + altitude
        self.position_vector = r * e_r

        # --- Cartesian velocity ---
        self.cart_velocity_vector = (
            radial_velocity * e_r +
            tangential_velocity * e_t
        )
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)

    def keplerian_initial_conditions(
            self,
            apogee,
            perigee,
            altitude,
            inclination,
            arg_perigee,
            raan,
            true_anomaly_sign=1,
            planet_radius=6_371_000.0,
            mu=3.986004418e14
    ):
        """
        apogee, perigee : m above surface
        altitude        : current altitude above surface (m)
        inclination     : rad
        arg_perigee     : rad
        raan            : rad (longitude of ascending node)
        true_anomaly_sign : +1 ascending, -1 descending
        planet_radius   : m
        mu              : m^3/s^2
        """

        # --- Convert to orbital radii ---
        ra = planet_radius + apogee
        rp = planet_radius + perigee
        r = planet_radius + altitude

        # --- Semi-major axis and eccentricity ---
        a = 0.5 * (ra + rp)
        e = (ra - rp) / (ra + rp)

        # --- True anomaly from conic equation ---
        cos_nu = (a * (1 - e ** 2) / r - 1) / e
        cos_nu = np.clip(cos_nu, -1.0, 1.0)
        nu = true_anomaly_sign * np.arccos(cos_nu)

        # --- Specific angular momentum ---
        h = np.sqrt(mu * a * (1 - e ** 2))

        # --- Position & velocity in perifocal frame ---
        r_pf = np.array([
            r * np.cos(nu),
            r * np.sin(nu),
            0.0
        ])

        v_pf = np.array([
            -mu / h * np.sin(nu),
            mu / h * (e + np.cos(nu)),
            0.0
        ])

        # --- Rotation matrices ---
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        # --- Perifocal → ECI transformation ---
        Q = R3(raan) @ R1(inclination) @ R3(arg_perigee)

        self.position_vector = Q @ r_pf
        self.cart_velocity_vector = Q @ v_pf
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)

        # Store orbital elements
        self.apogee = apogee
        self.perigee = perigee
        self.altitude = altitude
        self.inclination = inclination
        self.arg_perigee = arg_perigee
        self.raan = raan
        self.true_anomaly = nu

    def v_inertial_toSOG(self, v_in, r_vec=None):
        """
        Convert inertial velocity to speed over ground (SOG).

        Parameters
        ----------
        v_in : ndarray
            3D velocity in ECI (m/s)
        r_vec : ndarray
            3D position in ECI (m)

        Returns
        -------
        v_sog : float
            Speed over ground (m/s)
        """

        if r_vec is None:
            r_vec = self.position_vector

        # Earth's rotation rate (rad/s)
        omega = 2 * np.pi / 86164.0  # sidereal day

        # Earth rotation vector
        k_hat = np.array([0.0, 0.0, 1.0])

        # Velocity of ground due to rotation
        v_earth = omega * np.cross(k_hat, r_vec)

        # Relative velocity over ground
        v_sog_vec = v_in - v_earth

        return v_sog_vec

    def aero_accelerations(self, r_vec, v_vec, planet_radius=6_371_000.0, mu=3.986004418e14):
        """
        Returns the Cartesian acceleration vectors (ECI) due to:
            - Gravity
            - Drag
            - Lift

        Returns:
            a_g : gravity acceleration vector (m/s^2)
            a_d : drag acceleration vector (m/s^2)
            a_l : lift acceleration vector (m/s^2)
        """

        # --- Position & velocity ---
        # r_vec = self.position_vector
        # v_vec = self.cart_velocity_vector

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        v_hat = v_vec / v

        # --- Altitude above surface ---
        self.altitude = r - planet_radius

        # --- Atmospheric density from COESA-76 ---
        rho, temp, sound = atmospheric_properties(self.altitude)

        self.mach = v/sound

        # --- Gravity acceleration vector ---
        a_g = -mu / r ** 3 * r_vec

        # --- Drag acceleration vector ---

        self.dynamic_pressure = 0.5 * rho * v ** 2


        self.cl, self.cd = self.get_cl_cd(self.mach, self.alpha)
        # print(self.cl, self.cd)

        if np.isnan(self.dynamic_pressure):
            F_d=0.0
        else:
            F_d = self.dynamic_pressure * self.cd * self.Area
        a_d = -F_d / self.mass * v_hat

        # --- Lift acceleration vector ---
        # Local vertical
        r_hat = r_vec / r

        # Orbit normal
        h_hat = np.cross(r_hat, v_hat)
        h_norm = np.linalg.norm(h_hat)
        if h_norm < 1e-8:
            h_hat = np.array([0.0, 0.0, 1.0])  # avoid zero vector
        else:
            h_hat /= h_norm

        # Nominal lift direction (perpendicular to velocity)
        lift_hat_0 = np.cross(v_hat, h_hat)
        lift_hat_0 /= np.linalg.norm(lift_hat_0)

        # Rotate lift vector by banking angle about velocity
        phi = self.banking_angle
        lift_hat = np.cos(phi) * lift_hat_0 + np.sin(phi) * h_hat

        # Lift acceleration vector

        if np.isnan(self.dynamic_pressure):
            F_l=0.0
        else:
            F_l = self.dynamic_pressure * self.cl * self.Area
        a_l = F_l / self.mass * lift_hat

        self.a_g = a_g
        self.a_d = a_d
        self.a_l = a_l

        self.a = a_g + a_d + a_l

        # Stagnation heat flux (Sutton Grave's equation) https://tfaws.nasa.gov/TFAWS12/Proceedings/Aerothermodynamics%20Course.pdf

        k = 1.7415e-4  # (Earth)
        # k = 1.9027e-4  # (Mars)

        self.qs = k * np.sqrt(rho/self.nose_R) * v ** 3

        self.sog = v

        if np.isnan(self.qs):
            self.qs=0
        g0 = 9.80665  # m/s^2
        self.g = np.linalg.norm(self.a_l + self.a_d) / g0

        return self.a

    def Euler_Rich_step(self, dt=1):
        an = self.aero_accelerations(self.position_vector, self.cart_velocity_vector_og)
        vn = self.cart_velocity_vector
        yn = self.position_vector

        v_mid = vn + 0.5*dt*an
        y_mid = yn + 0.5*dt*vn


        a_mid = self.aero_accelerations(y_mid, self.v_inertial_toSOG(v_mid))

        v_next = vn + dt*a_mid
        y_next = yn + dt*v_mid

        # --- Local vertical unit vector (north-up) ---
        e_r = y_next / np.linalg.norm(y_next)  # points away from Earth's center

        # --- Descent rate in m/s ---
        self.descend_rate = -np.dot(v_next, e_r)  # positive when descending

        self.cart_velocity_vector = v_next
        self.position_vector = y_next
        self.cart_velocity_vector_og = self.v_inertial_toSOG(self.cart_velocity_vector)
        r_vec = self.position_vector
        v_vec = self.cart_velocity_vector

        r_hat = r_vec / np.linalg.norm(r_vec)  # local vertical unit vector
        v_radial = np.dot(v_vec, r_hat)  # radial component
        v_tan_vec = v_vec - v_radial * r_hat  # remove radial component
        self.v_tan = np.linalg.norm(v_tan_vec)  # tangential magnitude


    def banking_angle_dr_P_controller(self, targetDR, kP=1.5):
        self.targetDR = targetDR
        error =  self.descend_rate - targetDR
        altitude = self.altitude
        # We calculate here the orbital tangential velocity component v_tan
        a_req = kP * error + 9.81 - self.v_tan**2 / (altitude + 6371000)
        rho, temp, sound = atmospheric_properties(altitude)

        if np.isnan(rho):
            rho=0

        v = np.linalg.norm(self.cart_velocity_vector_og)
        available_acc = 0.5 * rho * v ** 2 * self.cl * self.Area / self.mass


        if a_req > available_acc:
            self.banking_angle = 0
        elif -a_req > available_acc:
            self.banking_angle = np.pi
        else:
            self.banking_angle = np.arccos(a_req/available_acc)

    def banking_angle_dr_PD_controller(self, targetDR, kP=1.5, kD=0.6):
        # Calculate error
        error = self.descend_rate - targetDR
        self.targetDR = targetDR

        # Derivative of error
        d_error = (error - getattr(self, 'prev_error_DR', 0)) / self.dt  # Use 0 if prev_error_DR not set

        # Save current error for next step
        self.prev_error_DR = error

        altitude = self.altitude
        # Orbital tangential acceleration component
        a_req = kP * error + kD * d_error + 9.81 - self.v_tan ** 2 / (altitude + 6371000)

        # Atmospheric properties
        rho, temp, sound = atmospheric_properties(altitude)
        if np.isnan(rho):
            rho = 0

        v = np.linalg.norm(self.cart_velocity_vector_og)
        available_acc = 0.5 * rho * v ** 2 * self.cl * self.Area / self.mass

        # Determine banking angle
        if a_req > available_acc:
            self.banking_angle = 0
        elif -a_req > available_acc:
            self.banking_angle = np.pi
        else:
            self.banking_angle = np.arccos(a_req / available_acc)

    def banking_angle_h_P_controller(self, target_altitude, kP_DR=1.5, kP_h = 0.02):

        self.target_altitude = target_altitude

        altitude_error = self.altitude - self.target_altitude
        DR = kP_h * altitude_error
        self.banking_angle_dr_PD_controller(DR, kP=kP_DR)

    def banking_angle_h_PD_controller(self, target_altitude, kP_DR=1.5, kD_DR=0.6, kP_h = 0.02, kD_h=0.94, max_DR=None):

        self.target_altitude = target_altitude



        altitude_error = self.altitude - self.target_altitude

        # Derivative of error
        # d_error = (altitude_error - getattr(self, 'prev_error_H', 0)) / self.dt  # Use 0 if prev_error_H not set
        d_error = self.descend_rate - getattr(self, 'targetDR', 0)

        # Save current error for next step
        self.prev_error_H = altitude_error
        if max_DR is None:
            DR = kP_h * altitude_error + kD_h * d_error
        else:
            DR = min(kP_h * altitude_error + kD_h * d_error, max_DR)
        self.banking_angle_dr_PD_controller(DR, kP=kP_DR, kD=kD_DR)

    def banking_angle_h_PD_controller_smart_glide(self, kP_DR=1.5, kP_h = 0.02):

        # Firstly I'll compute the 0 DR required density
        v = self.sog
        rho_req = self.mass*np.linalg.norm(self.a_g)/(0.5* v**2 * self.cl * self.Area)

        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        if self.target_altitude is None:
            if rho_req>1.22:
                DR=0
                self.targetDR = DR
                self.banking_angle = 0
            if rho_req<0.000001:
                DR=100
                self.banking_angle_dr_PD_controller(DR, kP=kP_DR)
        else:
            self.banking_angle_h_PD_controller(target_altitude=self.target_altitude, kP_DR=kP_DR)

    def banking_angle_h_PD_controller_smart_qc(self, kP_DR=1.5, kP_h=0.02, max_qcSF = 1, thresshold_alt=25_000.0, thresshold_g=2.5, start_offset_m=2000):
        # Firstly I'll compute the max qc required density
        k = 1.7415e-4  # (Earth)
        # k = 1.9027e-4  # (Mars)
        rho_req = self.nose_R * ((self.max_qc/max_qcSF) / (k * self.sog ** 3)) ** 2

        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        if self.target_altitude is None:
            if rho_req > 1.22:
                DR = 0
                self.targetDR = DR
                self.banking_angle = 0
            if rho_req < 0.000001:
                DR = 100
                self.banking_angle_dr_PD_controller(DR, kP=kP_DR)
        else:
            if thresshold_alt is not None:
                if self.target_altitude < thresshold_alt:
                    self.controller="PGC"


            if thresshold_g is not None:
                if self.g > thresshold_g:
                    self.controller = "PGC"

        if self.altitude - self.target_altitude < start_offset_m or self.started_c:
            self.started_c = True
            self.banking_angle_h_P_controller(target_altitude=self.target_altitude, kP_DR=kP_DR)
        else:
            self.targetDR = 35
            self.banking_angle_dr_P_controller(self.targetDR, kP=kP_DR)
    def banking_angle_h_PD_controller_smart_g_control(self, target_g=2.5, kP_DR=1.5, kP_h = 0.02):

        # Firstly I'll compute the required density to pull the target_gs
        v = self.sog
        cf = np.sqrt(self.cl**2 + self.cd**2)\
        # target_g = 0.5*rho_req* v**2 * cf * self.Area / self.mass
        g0 = 9.80665  # m/s^2
        rho_req = 2*target_g*g0*self.mass / (v**2 * cf * self.Area)
        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        if self.target_altitude is None:
            if rho_req>1.22:
                self.controller = "terminal"
            if rho_req<0.000001:
                DR=100
                self.banking_angle_dr_P_controller(DR, kP=kP_DR)
        else:
            if self.target_altitude<30000:
                self.controller = "terminal"
                self.alpha = 45.0
                self.max_banking_angle = np.pi/2
            else:
                self.banking_angle_h_P_controller(target_altitude=self.target_altitude, kP_DR=kP_DR)

    def get_cl_max_and_aoa_at_mach_interp(self, mach, n_aoa=300):
        """
        Returns:
            cl_max : maximum interpolated CL at given Mach
            aoa_at_cl_max : AOA (deg) where CL is maximum
        """

        # Clamp Mach to valid range
        mach = np.clip(mach, self.mach_min, self.mach_max)

        # Sweep AOA
        aoa_grid = np.linspace(self.aoa_min, self.aoa_max, n_aoa)
        mach_grid = np.full_like(aoa_grid, mach)

        CL = self._cl_interp(mach_grid, aoa_grid)

        # Mask invalid values
        valid = np.isfinite(CL)
        if not np.any(valid):
            return 0.0, self.alpha  # safe fallback

        CL_valid = CL[valid]
        aoa_valid = aoa_grid[valid]

        idx = np.argmax(CL_valid)

        cl_max = float(CL_valid[idx])
        aoa_at_cl_max = float(aoa_valid[idx])

        return cl_max, aoa_at_cl_max

    def solve_alpha_for_cl(
            self,
            mach,
            cl_target,
            cl_max,
            aoa_stall,
            branch="pre",  # "pre" or "post"
            n_aoa_scan=300,
            tol=1e-4
    ):
        """
        Solve for AOA (deg) such that:
            CL(mach, alpha) = cl_target

        Inputs:
            mach       : current Mach
            cl_target  : desired CL
            cl_max     : max CL at this Mach
            aoa_stall  : AOA where CL_max occurs

        Returns:
            alpha_deg or None if no valid solution
        """



        # Reject impossible request
        if cl_target < 0 or cl_target > cl_max:
            return None

        # Select AOA bounds
        if branch == "pre":
            aoa_lo = self.aoa_min
            aoa_hi = aoa_stall
        elif branch == "post":
            aoa_lo = aoa_stall
            aoa_hi = self.aoa_max
        else:
            raise ValueError("branch must be 'pre' or 'post'")

        aoa_lo = float(aoa_lo)
        aoa_hi = float(aoa_hi)

        # CL residual
        def f(alpha):
            cl, _ = self.get_cl_cd(mach, alpha)
            return float(cl) - cl_target

        aoa_scan = np.linspace(aoa_lo, aoa_hi, n_aoa_scan)

        f_scan = np.array(
            [f(a) for a in aoa_scan],
            dtype=float
        ).reshape(-1)  # <-- FORCE 1-D

        valid = np.isfinite(f_scan)

        aoa_scan = aoa_scan[valid]
        f_scan = f_scan[valid]

        if len(f_scan) < 2:
            return None

        idx = np.where(np.sign(f_scan[:-1]) != np.sign(f_scan[1:]))[0]
        if len(idx) == 0:
            return None

        a0, a1 = aoa_scan[idx[0]], aoa_scan[idx[0] + 1]

        # Root find
        try:
            alpha_sol = brentq(f, a0, a1, xtol=tol)
        except ValueError:
            return None

        return float(alpha_sol)

    def attack_angle_dr_PD_controller(self, targetDR, kP=1.5, kD=0.6):
        # Calculate error
        error = self.descend_rate - targetDR
        self.targetDR = targetDR

        # Derivative of error
        d_error = (error - getattr(self, 'prev_error_DR', 0)) / self.dt  # Use 0 if prev_error_DR not set

        # Save current error for next step
        self.prev_error_DR = error

        altitude = self.altitude
        # Orbital tangential acceleration component
        a_req = kP * error + kD * d_error + 9.81 - self.v_tan ** 2 / (altitude + 6371000)

        # Atmospheric properties
        rho, temp, sound = atmospheric_properties(altitude)
        if np.isnan(rho):
            rho = 0

        v = np.linalg.norm(self.cart_velocity_vector_og)
        mach = self.mach
        self.cl_max, opt_aoa = self.get_cl_max_and_aoa_at_mach_interp(mach) # get cl max at current mach
        available_acc = (0.5 * rho * v ** 2 * self.cl_max * self.Area / self.mass)*np.cos(self.banking_angle)
        inside_arccos=a_req/(0.5 * rho * v ** 2 * self.cl_max * self.Area / self.mass)

        if inside_arccos > 1:
            self.max_banking_angle=0
        elif inside_arccos<0:
            self.max_banking_angle=np.pi/2
        else:
            self.max_banking_angle = np.arccos(inside_arccos)

        # Determine attack angle
        if a_req > available_acc:
            self.alpha = opt_aoa
        elif a_req < 0:
            self.alpha = 89.9
        else:
            required_cl = (self.mass*a_req/(np.cos(self.banking_angle) * 0.5 * rho * v ** 2 * self.Area))
            self.alpha = self.solve_alpha_for_cl(
                mach=self.mach,
                cl_target=required_cl,
                cl_max=self.cl_max,
                aoa_stall=opt_aoa,
                branch="post"
            ) # find required alpha for required_cl

    def attack_angle_h_P_controller(self, target_altitude, kP_DR=1.5, kP_h = 0.02):

        self.target_altitude = target_altitude

        altitude_error = self.altitude - self.target_altitude
        DR = kP_h * altitude_error
        self.attack_angle_dr_PD_controller(DR, kP=kP_DR)

    def attack_angle_h_PD_controller(self, target_altitude, kP_DR=1.5, kD_DR=0.6, kP_h = 0.02, kD_h=0.94, max_DR=None):

        self.target_altitude = target_altitude



        altitude_error = self.altitude - self.target_altitude

        # Derivative of error
        # d_error = (altitude_error - getattr(self, 'prev_error_H', 0)) / self.dt  # Use 0 if prev_error_H not set
        d_error = self.descend_rate - getattr(self, 'targetDR', 0)

        # Save current error for next step
        self.prev_error_H = altitude_error
        if max_DR is None:
            DR = kP_h * altitude_error + kD_h * d_error
        else:
            DR = min(kP_h * altitude_error + kD_h * d_error, max_DR)
        self.attack_angle_dr_PD_controller(DR, kP=kP_DR, kD=kD_DR)

    def attack_angle_h_PD_controller_smart_glide(self, kP_DR=1.5, kP_h = 0.02):

        # Firstly I'll compute the 0 DR required density
        v = self.sog
        rho_req = self.mass*np.linalg.norm(self.a_g)/(0.5* v**2 * self.cl * self.Area)

        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        if self.target_altitude is None:
            if rho_req>1.22:
                DR=0
                self.targetDR = DR
                self.attack_angle_dr_PD_controller(DR, kP=kP_DR)
            if rho_req<0.000001:
                DR=100
                self.attack_angle_dr_PD_controller(DR, kP=kP_DR)
        else:
            self.attack_angle_h_PD_controller(target_altitude=self.target_altitude, kP_DR=kP_DR)

    def attack_angle_h_PD_controller_smart_qc(self, kP_DR=1.5, kP_h=0.02, max_qcSF = 1, thresshold_alt=25_000.0, thresshold_g=2.5, start_offset_m=2000):
        # Firstly I'll compute the max qc required density
        k = 1.7415e-4  # (Earth)
        # k = 1.9027e-4  # (Mars)
        rho_req = self.nose_R * ((self.max_qc/max_qcSF) / (k * self.sog ** 3)) ** 2

        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)



        if self.target_altitude is None:
            if rho_req > 1.22:
                DR = 0
                self.targetDR = DR
                self.attack_angle_dr_PD_controller(DR, kP=kP_DR)
            if rho_req < 0.000001:
                DR = 100
                self.attack_angle_dr_PD_controller(DR, kP=kP_DR)
        else:
            if thresshold_alt is not None:
                if self.target_altitude < thresshold_alt:
                    self.controller="aPGC"


            if thresshold_g is not None:
                if self.g > thresshold_g:
                    self.controller = "aPGC"

        if self.altitude - self.target_altitude < start_offset_m or self.started_c:
            self.started_c = True
            self.attack_angle_h_P_controller(target_altitude=self.target_altitude, kP_DR=kP_DR)
        else:
            self.targetDR = 35
            self.attack_angle_dr_PD_controller(self.targetDR, kP=kP_DR)

    def attack_angle_h_PD_controller_smart_g_control(self, target_g=2.5, kP_DR=1.5, kP_h = 0.02, cd_max=9):

        # Firstly I'll compute the required density to pull the target_gs
        v = self.sog
        cf = np.sqrt(self.cl_max**2 + cd_max**2)\
        # target_g = 0.5*rho_req* v**2 * cf * self.Area / self.mass
        g0 = 9.80665  # m/s^2
        rho_req = 2*target_g*g0*self.mass / (v**2 * cf * self.Area)
        # Then I find at what altitude do I get that density
        self.target_altitude = altitude_for_density(rho_req)




        if rho_req>1.2:
            DR=0
            self.targetDR = DR
            self.alpha = 45
            self.controller = "terminal"
        elif rho_req<0.000001:
            DR=100
            self.attack_angle_dr_PD_controller(DR, kP=kP_DR)
        else:
            self.attack_angle_h_P_controller(target_altitude=self.target_altitude, kP_DR=kP_DR, kP_h=kP_h)

    def get_heading_from_velocity(self):
        """
        Returns heading angle (deg from North, clockwise)
        computed from velocity over ground (SOG).
        """

        # Position unit vectors
        r = self.position_vector / np.linalg.norm(self.position_vector)

        # Earth-centered axes
        z_earth = np.array([0.0, 0.0, 1.0])

        # Local East and North unit vectors
        east = np.cross(z_earth, r)
        east /= np.linalg.norm(east)

        north = np.cross(r, east)
        north /= np.linalg.norm(north)

        # Velocity over ground
        v = self.cart_velocity_vector_og

        # Project velocity into horizontal plane
        v_north = np.dot(v, north)
        v_east = np.dot(v, east)

        # Heading: clockwise from North
        self.heading_rad = np.arctan2(v_east, v_north)

        self.heading_deg = np.degrees(self.heading_rad)
        if self.heading_deg < 0:
            self.heading_deg += 360.0



        return self.heading_deg

    def banking_angle_heading_PD_controller(self, target_heading, kP_la=14.0, kD_la = 0.5):
        current_heading = self.get_heading_from_velocity()
        heading_error = target_heading - current_heading
        required_acc = kP_la * heading_error
        altitude = self.altitude

        # Atmospheric properties
        rho, temp, sound = atmospheric_properties(altitude)
        if np.isnan(rho):
            rho = 0

        v = np.linalg.norm(self.cart_velocity_vector_og)
        mach = self.mach
        self.cl_max, opt_aoa = self.get_cl_max_and_aoa_at_mach_interp(mach)  # get cl max at current mach

        available_acc = (0.5 * rho * v ** 2 * self.cl_max * self.Area / self.mass)*np.sin(self.max_banking_angle)

        if required_acc > available_acc:
            self.banking_angle = -self.max_banking_angle
        elif required_acc < -available_acc:
            self.banking_angle = self.max_banking_angle
        else:
            self.banking_angle = -np.arcsin(required_acc/(0.5 * rho * v ** 2 * self.cl_max * self.Area / self.mass))


    def get_great_circle_heading_and_range(
            self,
            target_lat_deg,
            target_lon_deg,
            earth_radius=6371000.0
    ):
        """
        Returns:
            heading_deg : initial great-circle heading (deg from North, clockwise)
            range_m     : great-circle distance (meters)

        Uses current latitude/longitude stored in the object.
        """
        self.longitude, self.latitude = eci_to_lonlat(self.position_vector, self.t)

        # Current position (degrees → radians)
        lat1 = self.latitude
        lon1 = self.longitude

        # Target position
        lat2 = np.radians(target_lat_deg)
        lon2 = np.radians(target_lon_deg)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # --- Great-circle distance (haversine) ---
        a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )

        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        range_m = earth_radius * c

        # --- Initial great-circle heading ---
        y = np.sin(dlon) * np.cos(lat2)
        x = (
                np.cos(lat1) * np.sin(lat2)
                - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        )

        heading_rad = np.arctan2(y, x)
        heading_deg = np.degrees(heading_rad)

        if heading_deg < 0:
            heading_deg += 360.0

        return float(heading_deg), float(range_m)

    def direct_to_landing_heading_controller(self):
        self.target_heading, self.range = self.get_great_circle_heading_and_range(self.landing_lat, self.landing_lon)
        self.banking_angle_heading_PD_controller(self.target_heading)

    def banking_angle_range_S_turn_controller(
            self,
            heading_gain=-400  ,
            max_heading_offset_deg=15.0,
            min_heading_offset_deg=0.5,
            range_deadband=100.0,  # meters
            crossrange_limit=200_000.0,  # meters
            extra_range=17_000,  # meters
    ):
        """
        Shuttle-style range control using S-turns.

        Uses:
            - self.range              (actual remaining range)
            - self.range_interp()     (estimated remaining range)
            - self.banking_angle_heading_PD_controller()
        """

        # --- Required data ---
        if not hasattr(self, "range_interp"):
            raise RuntimeError("Remaining range map not loaded")

        self.target_heading, self.range = self.get_great_circle_heading_and_range(self.landing_lat, self.landing_lon)


        altitude = self.altitude
        speed = self.sog

        # --- Estimated remaining range ---
        est_range = self.range_interp(altitude, speed)
        if not np.isfinite(est_range):
            return  # fail-safe: do nothing

        actual_range = self.range + extra_range
        range_error = est_range - actual_range

        # --- Deadband: go straight when close ---
        if abs(range_error) < range_deadband:
            self.target_heading_cmd = self.target_heading
            self.banking_angle_heading_PD_controller(self.target_heading_cmd)
            return

        # --- Heading offset magnitude (shrink as range error shrinks) ---
        heading_offset = heading_gain * abs(range_error) / actual_range
        heading_offset = np.clip(
            heading_offset * max_heading_offset_deg,
            min_heading_offset_deg,
            max_heading_offset_deg
        )

        # --- Crossrange estimation ---
        # Shuttle logic: flip bank when crossrange exceeds limit
        if not hasattr(self, "s_turn_sign"):
            self.s_turn_sign = 1

        if abs(getattr(self, "crossrange", 0.0)) > crossrange_limit:
            self.s_turn_sign *= -1

        # --- Command heading ---
        self.target_heading_cmd = (
                self.target_heading
                + self.s_turn_sign * heading_offset
        )

        # Normalize heading
        self.target_heading_cmd %= 360.0

        # --- Execute heading controller ---
        self.banking_angle_heading_PD_controller(self.target_heading_cmd)

    def update_crossrange(self):
        """
        Estimates crossrange from great-circle track (meters).
        """

        # Current position
        lon, lat = self.last_lon, self.last_lat

        # Target
        lon_t = np.radians(self.landing_lon)
        lat_t = np.radians(self.landing_lat)

        # Bearing to target
        bearing, _ = self.get_great_circle_heading_and_range(
            self.landing_lat,
            self.landing_lon
        )
        bearing = np.radians(bearing)

        # Heading error
        hdg = np.radians(self.heading_deg)
        delta = hdg - bearing

        # Crossrange ≈ range * sin(heading error)
        self.crossrange = self.range * np.sin(delta)

    def run_reentry(self, gif=True, controller=None, plot=True, dt=0.1, planet_radius=6_371_000.0, mu=3.986004418e14, gif_name="reentry.gif", DTLH=False):
        """
        Simulates reentry until altitude < 1 km.
        Records:
            - Altitude
            - Speed
            - Mach
            - Banking angle
            - g-force
            - Descent rate
            - Position
        Generates:
            - Time-series plots (altitude, speed, Mach, banking, g-force, descent rate vs time)
            - Speed and Mach vs altitude
            - 3D trajectory plot
            - Animated GIF of graphs & trajectory
        """
        t = 0.0
        self.t=t
        self.started_c = False
        self.controller = controller
        self.max_banking_angle = 0.0
        initial_altitude = self.altitude
        # --- Initialization ---
        times = []
        altitudes = []
        speeds = []
        machs = []
        bank_angles = []
        max_bank_angles = []
        g_forces = []
        descent_rates = []
        positions = []
        target_altitudes = []
        targetDRs = []
        aoas = []
        heat_fluxes = []
        heat_loads = []
        sogs_vecs = []
        self.heading_deg = self.get_heading_from_velocity()
        self.target_heading, self.range = self.get_great_circle_heading_and_range(self.landing_lat, self.landing_lon)
        self.covered_distance = 0
        self.covered_distances = []
        self.distance_step = 0
        heat_load = 0.0  # J/m^2 (integral of heat flux)

        max_steps = 50000  # safeguard
        step = 0


        prev_altitude = initial_altitude

        self.dt = dt

        # Initialize once (before loop)
        if not hasattr(self, "log"):
            self.log = {
                "time": [],
                "heading": [],
                "gc_heading": [],
                "cmd_heading": [],
                "range": [],
                "est_range": [],
            }

        self.last_position_vector = self.position_vector

        while True:
            self.t = t
            r_vec = self.position_vector
            v_vec = self.cart_velocity_vector
            v_vec_og = self.cart_velocity_vector_og
            sogs_vecs.append(v_vec_og)
            altitude = np.linalg.norm(r_vec) - planet_radius

            self.last_lat, self.last_lon = self.latitude, self.longitude
            self.target_heading, self.range = self.get_great_circle_heading_and_range(self.landing_lat,
                                                                                      self.landing_lon)

            # Stop condition
            if altitude < 1000.0 or step >= max_steps:
                break

            # Take a step
            self.Euler_Rich_step(dt)

            # self.distance_step = short_great_circle_distance(self.last_lat, self.last_lon, self.latitude, self.longitude)

            self.distance_step = self.sog * dt

            self.covered_distance += self.distance_step


            self.covered_distances.append(self.covered_distance)

            # --- Thermal & aero quantities ---
            q_dyn = self.dynamic_pressure  # Pa
            q_dot = self.qs  # W/m^2 (stagnation heat flux)
            if q_dot > 0.0:
                heat_load += q_dot * dt  # J/m^2 (time integral)

            aoas.append(float(self.alpha))
            heat_fluxes.append(q_dot)
            heat_loads.append(heat_load)
            if hasattr(self, "max_banking_angle"):
                max_bank_angles.append(float(np.degrees(self.max_banking_angle)))
            else:
                max_bank_angles.append(0.0)

            if DTLH:
                # self.direct_to_landing_heading_controller()

                self.update_crossrange()
                self.banking_angle_range_S_turn_controller()

            if self.controller=="PDR":
                if altitude < 3000.0:
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    if altitude < 50000.0:
                        descend_rate = 0
                    else:
                        descend_rate = 5
                    self.banking_angle_dr_PD_controller(descend_rate)
            elif self.controller=="PH":
                if self.altitude < 3000.0:
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    self.banking_angle_h_PD_controller_smart_glide()
            elif self.controller=="PQC":
                if self.altitude < 3000.0:
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    self.banking_angle_h_PD_controller_smart_qc()
            elif self.controller=="PGC":
                # if self.altitude < 3000.0:
                #     self.banking_angle = 0
                #     self.alpha = 90
                # else:
                self.banking_angle_h_PD_controller_smart_g_control()
            elif self.controller=="aPDR":
                if altitude < 3000.0:
                    descend_rate = 0
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    descend_rate = 150
                    self.attack_angle_dr_PD_controller(descend_rate)
            elif self.controller=="aPH":
                if self.altitude < 3000.0:
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    self.attack_angle_h_PD_controller_smart_glide()
            elif self.controller=="aPQC":
                if self.altitude < 3000.0:
                    self.banking_angle = 0
                    self.alpha = 90
                else:
                    self.attack_angle_h_PD_controller_smart_qc()
            elif self.controller=="aPGC":
                # if self.altitude < 7000.0:
                #     self.banking_angle = 0
                #     self.alpha = 90
                # else:
                if altitude < 35000.0:
                    self.controller = "terminal"
                else:
                    self.attack_angle_h_PD_controller_smart_g_control()
            elif self.controller=="terminal":
                if self.altitude < 2000.0 or self.range<1000.0 or self.target_heading>200.0:
                    self.banking_angle = 0
                    self.alpha = 90
                    DTLH=False
                else:
                    self.alpha = 45
                    self.max_banking_angle = np.pi/2
                    self.targetDR=0
                    self.target_altitude = 0



            # Compute descent rate
            descent_rate = self.descend_rate

            # Record state
            times.append(t)
            altitudes.append(altitude)
            speed = self.sog
            speeds.append(speed)
            machs.append(self.mach)
            bank_angles.append(np.rad2deg(self.banking_angle))  # degrees

            g_forces.append(self.g)
            descent_rates.append(descent_rate)
            positions.append(r_vec.copy())
            # Record target values if available
            if hasattr(self, 'target_altitude') and self.target_altitude is not None:
                target_altitudes.append(self.target_altitude)
            else:
                target_altitudes.append(np.nan)

            if hasattr(self, 'targetDR') and self.targetDR is not None:
                targetDRs.append(self.targetDR)
            else:
                targetDRs.append(np.nan)

            # Each timestep
            gc_heading, _ = self.get_great_circle_heading_and_range(
                self.landing_lat,
                self.landing_lon
            )

            est_range = (
                self.range_interp(self.altitude, self.sog)
                if hasattr(self, "range_interp") else np.nan
            )

            self.log["time"].append(t)
            self.log["heading"].append(self.heading_deg)
            self.log["gc_heading"].append(gc_heading)
            self.log["cmd_heading"].append(getattr(self, "target_heading_cmd", gc_heading))
            self.log["range"].append(self.range)
            self.log["est_range"].append(est_range)

            # Update time
            t += dt
            step += 1

            percent_down = 100 * (initial_altitude - altitude) / initial_altitude

            print(
                f"\rTime: {t:.1f}s | "
                f"Range: {float(self.range / 1000):.2f} km | "
                f"Covered: {float(self.covered_distance / 1000):.2f} km | "
                f"Head: {self.heading_deg:.2f} ° | "
                f"Heading target: {float(self.target_heading):.2f} ° | "
                f"Heading error: {float(self.target_heading - self.heading_deg):.2f} ° | "
                f"Alt: {altitude / 1000:.2f} km | "
                f"Mach: {self.mach:.2f} | "
                f"Bank: {float(np.rad2deg(self.banking_angle)):.1f}° | "
                f"{percent_down:.2f}% down",
                end='', flush=True
            )

        if len(times) == 0:
            self.times = np.array([])
            self.altitudes = np.array([])
            self.speeds = np.array([])
            self.machs = np.array([])
            self.bank_angles = np.array([])
            self.g_forces = np.array([])
            self.descent_rates = np.array([])
            self.positions = np.empty((0, 3))
            self.target_altitudes = np.array([])
            self.targetDRs = np.array([])
            self.aoas = np.array([])
            self.heat_fluxes = np.array([])
            self.heat_loads = np.array([])
            self.max_bank_angles = np.array([])
            self.sogs_vecs = np.empty((0, 3))

            return (
                self.times,
                self.altitudes,
                self.speeds,
                self.machs,
                self.bank_angles,
                self.g_forces,
                self.descent_rates,
                self.positions,
                np.array([]),  # lon
                np.array([]),  # lat
                self.heat_loads,
                self.heat_fluxes,
                self.aoas,
                self.sogs_vecs,
            )

        # Convert to arrays
        self.times = np.array(times)
        self.altitudes = np.array(altitudes)
        self.speeds = np.array(speeds)
        self.machs = np.array(machs)
        self.bank_angles = np.array(bank_angles)
        self.g_forces = np.array(g_forces)
        self.descent_rates = np.array(descent_rates)
        self.positions = np.array(positions)  # shape: (N, 3)
        self.target_altitudes = np.array(target_altitudes)
        self.targetDRs = np.array(targetDRs)
        self.aoas = np.array(aoas)
        self.heat_fluxes = np.array(heat_fluxes)
        self.heat_loads = np.array(heat_loads)
        self.max_bank_angles = np.array(max_bank_angles)
        self.covered_distances = np.array(self.covered_distances)
        self.sogs_vecs = np.array(sogs_vecs)

        if plot:

            self.plot_guidance_summary()
            # Create figure with gridspec for 2D + larger ground track
            fig = plt.figure(figsize=(20, 28))
            # Last row larger, top rows smaller to center the ground track vertically
            gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 3])

            # 2D subplots
            ax_alt = fig.add_subplot(gs[0, 0])
            ax_speed = fig.add_subplot(gs[0, 1])
            ax_mach = fig.add_subplot(gs[1, 0])
            ax_speed_alt = fig.add_subplot(gs[1, 1])
            ax_mach_alt = fig.add_subplot(gs[2, 0])
            ax_bank = fig.add_subplot(gs[2, 1])
            ax_g = fig.add_subplot(gs[3, 0])
            ax_descent = fig.add_subplot(gs[3, 1])
            ax_aoas = fig.add_subplot(gs[4, 0])
            ax_heat = fig.add_subplot(gs[4, 1])
            ax_heat_load = ax_heat.twinx()

            # Last row: ground track spanning full width
            ax_gt = fig.add_subplot(gs[5, :], projection=ccrs.PlateCarree())

            # --- Altitude vs Time ---
            ax_alt.plot(self.times, self.altitudes, 'r', label="Altitude")
            if np.any(~np.isnan(self.target_altitudes)):
                ax_alt.plot(self.times, self.target_altitudes, 'r--', label="Target Altitude")
            ax_alt.set_xlabel("Time (s)")
            ax_alt.set_ylabel("Altitude (m)")
            ax_alt.set_title("Altitude vs Time")
            ax_alt.legend()

            # --- Speed vs Time ---
            ax_speed.plot(self.times, self.speeds, 'b')
            ax_speed.set_xlabel("Time (s)")
            ax_speed.set_ylabel("Speed (m/s)")
            ax_speed.set_title("Speed vs Time")

            # --- Mach vs Time ---
            ax_mach.plot(self.times, self.machs, 'g')
            ax_mach.set_xlabel("Time (s)")
            ax_mach.set_ylabel("Mach")
            ax_mach.set_title("Mach vs Time")

            # --- Speed vs Altitude ---
            ax_speed_alt.plot(self.altitudes, self.speeds, 'b')
            ax_speed_alt.set_xlabel("Altitude (m)")
            ax_speed_alt.set_ylabel("Speed (m/s)")
            ax_speed_alt.set_title("Speed vs Altitude")

            # --- Mach vs Altitude ---
            ax_mach_alt.plot(self.altitudes, self.machs, 'g')
            ax_mach_alt.set_xlabel("Altitude (m)")
            ax_mach_alt.set_ylabel("Mach")
            ax_mach_alt.set_title("Mach vs Altitude")

            # --- Banking vs Time ---
            ax_bank.plot(self.times, self.bank_angles, 'm')
            ax_bank.plot(self.times, self.max_bank_angles, 'm--')
            ax_bank.plot(self.times, -self.max_bank_angles, 'm--')
            ax_bank.set_xlabel("Time (s)")
            ax_bank.set_ylabel("Bank (deg)")
            ax_bank.set_title("Banking Angle vs Time")

            # --- g-force vs Time ---
            ax_g.plot(self.times, self.g_forces, 'c')
            ax_g.set_xlabel("Time (s)")
            ax_g.set_ylabel("g")
            ax_g.set_title("g-force vs Time")

            # --- Descent Rate vs Time ---
            ax_descent.plot(self.times, self.descent_rates, 'k', label="Descent Rate")
            if np.any(~np.isnan(self.targetDRs)):
                ax_descent.plot(self.times, self.targetDRs, 'k--', label="Target DR")
            ax_descent.set_xlabel("Time (s)")
            ax_descent.set_ylabel("Descent rate (m/s)")
            ax_descent.set_title("Descent Rate vs Time")
            ax_descent.legend()

            ax_aoas.plot(self.times, self.aoas, color="orange")
            ax_aoas.set_xlabel("Time (s)")
            ax_aoas.set_ylabel("Angle of Attack (deg)")
            ax_aoas.set_title("Angle of Atack vs Time")

            # Heat flux (left axis)
            ax_heat.plot(
                times,
                self.heat_fluxes / 1e4,
                color="red",
                label="Heat Flux"
            )

            # Heat load (right axis)
            ax_heat_load.plot(
                times,
                self.heat_loads / 1e7,
                color="black",
                label="Heat Load"
            )

            ax_heat.fill_between(
                times,
                0,
                self.heat_fluxes / 1e4,
                color="red",
                alpha=0.25
            )

            ax_heat.set_xlabel("Time (s)")
            ax_heat.set_ylabel("Heat Flux (×1e4 W/m²)", color="red")
            ax_heat_load.set_ylabel("Heat Load (×1e7 J/m²)", color="black")

            ax_heat.set_title("Heat Flux & Integrated Heat Load vs Time")

            # --- Max heat flux ---
            idx_qdot = np.argmax(self.heat_fluxes)
            ax_heat.annotate(
                f"Max Heat Flux\n{self.heat_fluxes[idx_qdot]:.2e} W/m²",
                xy=(self.times[idx_qdot], self.heat_fluxes[idx_qdot] / 1e4),
                xytext=(10, 20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red"
            )

            # --- Horizontal line at max heat flux ---
            ax_heat.axhline(
                self.max_qc / 1e4,
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.8
            )

            # --- Max heat load ---
            idx_qload = np.argmax(self.heat_loads)
            ax_heat_load.annotate(
                f"Max Heat Load\n{self.heat_loads[idx_qload]:.2e} J/m²",
                xy=(self.times[idx_qload], self.heat_loads[idx_qload] / 1e7),
                xytext=(-120, -30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="black"),
                color="black"
            )

            # --- Ground Track with altitude color ---
            lon, lat = eci_to_lonlat(self.positions, self.times)
            sc = ax_gt.scatter(
                np.rad2deg(lon), np.rad2deg(lat),
                c=altitudes, cmap='plasma', s=1,
                transform=ccrs.Geodetic()
            )
            ax_gt.stock_img()
            ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_gt.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax_gt.add_feature(cfeature.COASTLINE)
            ax_gt.set_title("Ground Track with Altitude")
            ax_gt.set_xlabel("Longitude (deg)")
            ax_gt.set_ylabel("Latitude (deg)")

            # Plot landing site
            ax_gt.plot(
                self.landing_lon,
                self.landing_lat,
                marker=".",
                markersize=2,
                color="red",
                transform=ccrs.PlateCarree(),
                label="Landing Site"
            )

            ax_gt.legend(loc="upper right")

            # Colorbar for altitude
            cbar = plt.colorbar(sc, ax=ax_gt, orientation='vertical', fraction=0.03, pad=0.02)
            cbar.set_label("Altitude (m)")

            plt.tight_layout()
            plt.show()

        # ----------------------------
        # GIF Animation
        # ----------------------------
        if gif:
            max_frames = 50
            fps = 20
            interval = 50

            total_steps = len(times)
            if total_steps <= max_frames:
                frame_indices = np.arange(total_steps)
            else:
                frame_indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)

            # --- Figure and GridSpec like the static plot ---
            # Create figure with gridspec for 2D + larger ground track
            fig = plt.figure(figsize=(20, 28))
            # Last row larger, top rows smaller to center the ground track vertically
            gs = gridspec.GridSpec(6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 3])

            ax_alt = fig.add_subplot(gs[0, 0])
            ax_speed = fig.add_subplot(gs[0, 1])
            ax_mach = fig.add_subplot(gs[1, 0])
            ax_speed_alt = fig.add_subplot(gs[1, 1])
            ax_mach_alt = fig.add_subplot(gs[2, 0])
            ax_bank = fig.add_subplot(gs[2, 1])
            ax_g = fig.add_subplot(gs[3, 0])
            ax_descent = fig.add_subplot(gs[3, 1])
            ax_aoas = fig.add_subplot(gs[4, 0])
            ax_heat = fig.add_subplot(gs[4, 1])
            ax_heat_load = ax_heat.twinx()

            # --- Max heat flux reference line ---
            qdot_max = self.max_qc / 1e4
            ax_heat.axhline(
                qdot_max,
                color="red",
                linestyle=":",
                linewidth=2,
                alpha=0.8
            )

            ax_gt = fig.add_subplot(gs[5, :], projection=ccrs.PlateCarree())

            # --- Pre-setup ground track ---
            lon, lat = eci_to_lonlat(positions, times)
            sc = ax_gt.scatter([], [], c=[], cmap='plasma', s=30, transform=ccrs.Geodetic())
            ax_gt.stock_img()
            ax_gt.add_feature(cfeature.LAND, facecolor='lightgray')
            ax_gt.add_feature(cfeature.OCEAN, facecolor='lightblue')
            ax_gt.add_feature(cfeature.COASTLINE)
            ax_gt.set_title("Ground Track with Altitude")
            ax_gt.set_xlabel("Longitude (deg)")
            ax_gt.set_ylabel("Latitude (deg)")
            cbar = plt.colorbar(sc, ax=ax_gt, orientation='vertical', fraction=0.03, pad=0.02)
            cbar.set_label("Altitude (m)")

            # --- Empty lines for animation ---
            line_alt, = ax_alt.plot([], [], 'r')
            line_speed, = ax_speed.plot([], [], 'b')
            line_mach, = ax_mach.plot([], [], 'g')
            line_speed_alt, = ax_speed_alt.plot([], [], 'b')
            line_mach_alt, = ax_mach_alt.plot([], [], 'g')
            line_bank, = ax_bank.plot([], [], 'm')
            line_g, = ax_g.plot([], [], 'c')
            line_descent, = ax_descent.plot([], [], 'k')
            line_aoas, = ax_aoas.plot([], [], color="orange")
            line_qdot, = ax_heat.plot([], [], color="red")
            line_qload, = ax_heat_load.plot([], [], color="black")

            def update(frame_idx):
                frame = frame_indices[frame_idx]
                print(f"\rRendering GIF frame {frame_idx + 1}/{len(frame_indices)}...", end='', flush=True)

                # 2D plots
                line_alt.set_data(times[:frame], altitudes[:frame])
                ax_alt.relim();
                ax_alt.autoscale_view()
                line_speed.set_data(times[:frame], speeds[:frame])
                ax_speed.relim();
                ax_speed.autoscale_view()
                line_mach.set_data(times[:frame], machs[:frame])
                ax_mach.relim();
                ax_mach.autoscale_view()
                line_speed_alt.set_data(altitudes[:frame], speeds[:frame])
                ax_speed_alt.relim();
                ax_speed_alt.autoscale_view()
                line_mach_alt.set_data(altitudes[:frame], machs[:frame])
                ax_mach_alt.relim();
                ax_mach_alt.autoscale_view()
                line_bank.set_data(times[:frame], bank_angles[:frame])
                ax_bank.relim();
                ax_bank.autoscale_view()
                line_g.set_data(times[:frame], g_forces[:frame])
                ax_g.relim();
                ax_g.autoscale_view()
                line_descent.set_data(times[:frame], descent_rates[:frame])
                ax_descent.relim();
                ax_descent.autoscale_view()
                line_aoas.set_data(times[:frame], aoas[:frame] / 1e3)
                ax_aoas.relim()
                ax_aoas.autoscale_view()

                line_qdot.set_data(
                    times[:frame],
                    heat_fluxes[:frame] / 1e4
                )

                line_qload.set_data(
                    times[:frame],
                    heat_loads[:frame] / 1e7
                )


                ax_heat.relim()
                ax_heat.autoscale_view()

                ax_heat_load.relim()
                ax_heat_load.autoscale_view()

                idx_qdot = np.argmax(heat_fluxes)
                idx_qload = np.argmax(heat_loads)

                ax_heat.annotate(
                    "Max Heat Flux",
                    xy=(times[idx_qdot], heat_fluxes[idx_qdot] / 1e4),
                    xytext=(10, 20),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red"
                )

                ax_heat_load.annotate(
                    "Max Heat Load",
                    xy=(times[idx_qload], heat_loads[idx_qload] / 1e7),
                    xytext=(-120, -30),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    color="black"
                )

                # Ground track
                sc.set_offsets(np.column_stack((np.rad2deg(lon[:frame]), np.rad2deg(lat[:frame]))))
                sc.set_array(altitudes[:frame])

                return (line_alt, line_speed, line_mach,
                        line_speed_alt, line_mach_alt, line_bank,
                        line_g, line_descent, sc, line_aoas, line_qdot, line_qload)

            anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval, blit=False)
            # writer = PillowWriter(fps=fps) # Pillow
            writer = FFMpegWriter(fps=20) # FFMpeg

            anim.save("reentry.mp4", writer=writer) # FFMpeg
            # anim.save(gif_name, writer=writer) # Pillow

            print("\r" + " " * 120 + "\r", end='')  # clear loading line
            print(f"Reentry animation saved as {gif_name}")

        lon, lat = eci_to_lonlat(self.positions, self.times)

        return self.times, self.altitudes, self.speeds, self.machs, self.bank_angles, self.g_forces, self.descent_rates, self.positions, lon, lat, self.heat_loads, self.heat_fluxes, self.aoas, self.sogs_vecs

    def plot_guidance_summary(self):
        import matplotlib.pyplot as plt
        import numpy as np

        t = np.array(self.log["time"])
        heading = np.unwrap(np.radians(self.log["heading"])) * 180 / np.pi
        gc_heading = np.unwrap(np.radians(self.log["gc_heading"])) * 180 / np.pi
        cmd_heading = np.unwrap(np.radians(self.log["cmd_heading"])) * 180 / np.pi

        rng = np.array(self.log["range"])
        est_rng = np.array(self.log["est_range"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Heading plot ---
        ax1.plot(t, heading, label="Actual Heading")
        ax1.plot(t, gc_heading, "--", label="Great-Circle Heading")
        ax1.plot(t, cmd_heading, "-.", label="Commanded Heading")

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Heading [deg]")
        ax1.set_title("Heading vs Target")
        ax1.legend()
        ax1.grid(True)

        # --- Range plot ---

        ax2.plot(t, rng / 1000, label="Actual Range to Target")
        ax2.plot(t, est_rng / 1000, "--", label="Estimated Remaining Range")

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Range [km]")
        ax2.set_title("Range Management")

        range_error = est_rng - rng
        ax2b = ax2.twinx()
        ax2b.plot(t, range_error / 1000, ":", label="Range Error")
        ax2b.set_ylabel("Range Error [km]")

        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def build_remaining_range_map_aPQC(
            self,
            bank_angles_deg,
            dt=0.1,
            planet_radius=6_371_000.0,
            save_prefix="remaining_range_map"
    ):
        """
        Builds remaining-path-distance interpolator using run_reentry().

        remaining_range = f(altitude, speed_over_ground)

        NOTE:
        Remaining range here is INTENTIONAL path distance (not geometric range).
        """

        from scipy.interpolate import LinearNDInterpolator
        import numpy as np
        import matplotlib.pyplot as plt

        all_alt = []
        all_speed = []
        all_remaining = []

        # --- Save initial vehicle state ---
        r0 = self.position_vector.copy()
        v0 = self.cart_velocity_vector.copy()
        v0_og = self.cart_velocity_vector_og.copy()
        alpha0 = self.alpha

        for bank_deg in bank_angles_deg:
            print(f"\nRunning aPQC reentry | fixed bank = {bank_deg:.1f} deg")

            # --- Reset state ---
            self.position_vector = r0.copy()
            self.cart_velocity_vector = v0.copy()
            self.cart_velocity_vector_og = v0_og.copy()
            self.alpha = alpha0

            self.banking_angle = np.deg2rad(bank_deg)

            # IMPORTANT: reset path-distance bookkeeping
            self.covered_distance = 0.0
            self.covered_distances = []

            # --- Run reentry ---
            self.run_reentry(
                gif=False,
                plot=False,
                controller="aPQC",
                dt=dt,
                planet_radius=planet_radius,
                DTLH=False
            )

            # --- Extract histories ---
            alt = np.asarray(self.altitudes)
            speed = np.asarray(self.speeds)  # MUST be SOG
            covered = np.asarray(self.covered_distances)  # MUST be SOG-integrated

            n = min(len(alt), len(speed), len(covered))
            alt = alt[:n]
            speed = speed[:n]
            covered = covered[:n]

            if n < 10:
                continue  # discard broken runs

            total_path_length = covered[-1]
            remaining_path = total_path_length - covered

            # --- Filter garbage ---
            valid = (
                    np.isfinite(alt) &
                    np.isfinite(speed) &
                    np.isfinite(remaining_path) &
                    (remaining_path >= 0.0)
            )

            all_alt.append(alt[valid])
            all_speed.append(speed[valid])
            all_remaining.append(remaining_path[valid])

        # --- Concatenate all trajectories ---
        all_alt = np.concatenate(all_alt)
        all_speed = np.concatenate(all_speed)
        all_remaining = np.concatenate(all_remaining)

        # --- Build interpolator ---
        self.range_interp = LinearNDInterpolator(
            np.column_stack((all_alt, all_speed)),
            all_remaining
        )

        # --- Save dataset (CONSISTENT KEYS) ---
        np.savez(
            f"{save_prefix}.npz",
            altitude=all_alt,
            speed=all_speed,
            remaining_range=all_remaining
        )

        # --- Diagnostic plot ---
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(
            all_speed / 1000,
            all_alt / 1000,
            c=all_remaining / 1000,
            s=4
        )
        plt.colorbar(sc, label="Remaining Path Distance (km)")
        plt.xlabel("Speed over Ground (km/s)")
        plt.ylabel("Altitude (km)")
        plt.title("Remaining Path Distance Map – aPQC")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"\nSaved remaining-range dataset to {save_prefix}.npz")

        return self.range_interp

    def remaining_range_safe(self, altitude, speed):
        """
        Safe evaluation of remaining-range interpolator.
        Returns 0.0 if outside interpolation domain or invalid.
        """
        if not hasattr(self, "range_interp") or self.range_interp is None:
            return 0.0

        val = self.range_interp(altitude, speed)

        if val is None:
            return 0.0

        val = np.asarray(val)

        if not np.isfinite(val):
            return 0.0

        return float(val)

    def plot_remaining_range_interpolation(
            self,
            alt_min=0.0,
            alt_max=120_000.0,
            speed_min=500.0,
            speed_max=8000.0,
            n_alt=200,
            n_speed=200
    ):
        """
        Plots continuous remaining-range interpolation using safe evaluation.
        """

        if not hasattr(self, "range_interp"):
            raise RuntimeError("Run build_remaining_range_map_aPQC() first")

        alt_grid = np.linspace(alt_min, alt_max, n_alt)
        speed_grid = np.linspace(speed_min, speed_max, n_speed)

        ALT, SPEED = np.meshgrid(alt_grid, speed_grid)

        RANGE = np.zeros_like(ALT)

        for i in range(n_speed):
            for j in range(n_alt):
                RANGE[i, j] = self.remaining_range_safe(
                    ALT[i, j],
                    SPEED[i, j]
                )

        # Mask zero values (outside domain)
        RANGE_MASKED = np.ma.masked_where(RANGE <= 0.0, RANGE)

        plt.figure(figsize=(10, 7))
        im = plt.pcolormesh(
            SPEED / 1000.0,
            ALT / 1000.0,
            RANGE_MASKED / 1000.0,
            shading="auto",
            cmap="viridis"
        )

        plt.colorbar(im, label="Remaining Range (km)")
        plt.xlabel("Speed over Ground (km/s)")
        plt.ylabel("Altitude (km)")
        plt.title("Remaining Range Interpolation – aPQC")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_orbit_3d_init(
            self,
            apogee,
            perigee,
            inclination,
            arg_perigee,
            raan,
            planet_radius=6_371_000.0,
            mu=3.986004418e14,
            num_points=500,
            velocity_scale=0.1
    ):
        """
        Plots the full orbit, spacecraft position, and velocity vector in 3D (ECI frame)

        velocity_scale : fraction of planet radius used for arrow length
        """

        # --- Orbital parameters ---
        ra = planet_radius + apogee
        rp = planet_radius + perigee
        a = 0.5 * (ra + rp)
        e = (ra - rp) / (ra + rp)

        # True anomaly range
        nu = np.linspace(0, 2 * np.pi, num_points)

        # Radius as function of true anomaly
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))

        # Perifocal coordinates
        r_pf = np.vstack((
            r * np.cos(nu),
            r * np.sin(nu),
            np.zeros_like(nu)
        ))

        # Rotation matrices
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        # Perifocal → ECI transformation
        Q = R3(raan) @ R1(inclination) @ R3(arg_perigee)
        r_eci = Q @ r_pf

        # --- Plot ---
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Planet
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        x = planet_radius * np.cos(u) * np.sin(v)
        y = planet_radius * np.sin(u) * np.sin(v)
        z = planet_radius * np.cos(v)
        ax.plot_surface(x, y, z, alpha=0.3)

        # Orbit
        ax.plot(
            r_eci[0], r_eci[1], r_eci[2],
            label="Orbit"
        )

        # Spacecraft position
        r_sc = self.position_vector
        v_sc = self.cart_velocity_vector

        ax.scatter(
            r_sc[0], r_sc[1], r_sc[2],
            s=60,
            label="Spacecraft"
        )

        # --- Velocity vector ---
        v_hat = v_sc / np.linalg.norm(v_sc)
        arrow_length = velocity_scale * planet_radius

        ax.quiver(
            r_sc[0], r_sc[1], r_sc[2],
            v_hat[0], v_hat[1], v_hat[2],
            length=arrow_length,
            normalize=True,
            linewidth=2,
            label="Velocity"
        )

        # Formatting
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Orbit, Spacecraft Position, and Velocity Vector")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        plt.show()

    def plot_orbit_3d(
            self,
            planet_radius=6_371_000.0,
            mu=3.986004418e14,
            num_points=500,
            velocity_scale=0.1
    ):
        """
        Plots the orbit using ONLY the current Cartesian state:
        - self.position_vector (ECI)
        - self.cart_velocity_vector (ECI)
        """

        import numpy as np
        import matplotlib.pyplot as plt

        r_vec = self.position_vector
        v_vec = self.cart_velocity_vector

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # ----------------------------
        # Orbital elements from state
        # ----------------------------
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
        e = np.linalg.norm(e_vec)

        a = 1 / (2 / r - v ** 2 / mu)

        i = np.arccos(h_vec[2] / h)

        K = np.array([0.0, 0.0, 1.0])
        n_vec = np.cross(K, h_vec)
        n = np.linalg.norm(n_vec)

        raan = np.arctan2(n_vec[1], n_vec[0])

        arg_perigee = np.arccos(
            np.dot(n_vec, e_vec) / (n * e)
        ) if e > 1e-8 else 0.0

        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee

        # ----------------------------
        # Reconstruct orbit
        # ----------------------------
        nu = np.linspace(0, 2 * np.pi, num_points)
        p = a * (1 - e ** 2)
        r_orbit = p / (1 + e * np.cos(nu))

        r_pf = np.vstack((
            r_orbit * np.cos(nu),
            r_orbit * np.sin(nu),
            np.zeros_like(nu)
        ))

        # Rotation matrices
        def R3(theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        def R1(theta):
            return np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])

        Q = R3(raan) @ R1(i) @ R3(arg_perigee)
        r_eci = Q @ r_pf

        # ----------------------------
        # Plot
        # ----------------------------
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")

        # Planet
        u_sph, v_sph = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        x = planet_radius * np.cos(u_sph) * np.sin(v_sph)
        y = planet_radius * np.sin(u_sph) * np.sin(v_sph)
        z = planet_radius * np.cos(v_sph)
        ax.plot_surface(x, y, z, alpha=0.3)

        # Orbit
        ax.plot(r_eci[0], r_eci[1], r_eci[2], label="Orbit")

        # Spacecraft position
        ax.scatter(
            r_vec[0], r_vec[1], r_vec[2],
            s=60, label="Spacecraft"
        )

        # Velocity vector
        v_hat = v_vec / v
        arrow_length = velocity_scale * planet_radius

        ax.quiver(
            r_vec[0], r_vec[1], r_vec[2],
            v_hat[0], v_hat[1], v_hat[2],
            length=arrow_length,
            normalize=True,
            linewidth=2,
            label="Velocity"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Orbit from Cartesian State (ECI)")
        ax.legend()
        ax.set_box_aspect([1, 1, 1])

        # ----------------------------
        # Banking (lift) vector
        # ----------------------------
        r_hat = r_vec / np.linalg.norm(r_vec)
        v_hat = v_vec / np.linalg.norm(v_vec)

        h_hat = np.cross(r_hat, v_hat)
        h_hat /= np.linalg.norm(h_hat)

        lift_hat_0 = np.cross(v_hat, h_hat)
        lift_hat_0 /= np.linalg.norm(lift_hat_0)

        phi = self.banking_angle  # radians

        lift_hat = (
                np.cos(phi) * lift_hat_0 +
                np.sin(phi) * h_hat
        )

        lift_length = 0.08 * planet_radius

        ax.quiver(
            r_vec[0], r_vec[1], r_vec[2],
            lift_hat[0], lift_hat[1], lift_hat[2],
            length=lift_length,
            normalize=True,
            linewidth=2,
            linestyle="dashed",
            label="Bank / Lift",
            color="green"
        )

        plt.show()

    def load_remaining_range_map(self, filename):
        """
        Load remaining-range interpolation map from a saved .npz file.

        Expected keys in file:
            - altitude : meters
            - speed    : m/s (speed over ground)
            - range    : meters (remaining range)

        Builds:
            self.range_interp(altitude, speed) -> remaining_range
        """

        from scipy.interpolate import LinearNDInterpolator

        data = np.load(filename)

        # Required fields
        alt = data["altitude"].astype(float)
        speed = data["speed"].astype(float)
        remaining_range = data["remaining_range"].astype(float)

        # Basic sanity checks
        valid = (
                np.isfinite(alt) &
                np.isfinite(speed) &
                np.isfinite(remaining_range)
        )

        if not np.any(valid):
            raise ValueError("Remaining range map contains no valid data")

        alt = alt[valid]
        speed = speed[valid]
        remaining_range = remaining_range[valid]

        # Store raw data (useful for debugging / plotting)
        self.range_map_altitude = alt
        self.range_map_speed = speed
        self.range_map_remaining = remaining_range

        # Build interpolator
        self.range_interp = LinearNDInterpolator(
            np.column_stack((alt, speed)),
            remaining_range
        )

        self.remaining_range_loaded = True


