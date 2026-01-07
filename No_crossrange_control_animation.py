import reentripy as rpy
import numpy as np
from pystdatm import density

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from panda3d.core import load_prc_file
import simplepbr
from panda3d.core import DirectionalLight, AmbientLight, Vec3, Vec4


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

# times, altitudes, speeds, machs, bank_angles, g_forces, descent_rates, positions, lon, lat, heat_loads, heat_fluxes, aoas, sogs_vecs = sc.run_reentry(gif=False, controller="aPQC", DTLH=True) # With controller adjusting bank trying to keep max heating (DOESN'T REALLY WORK --> HIGH Gs)

from panda3d.core import LineSegs, NodePath





class Animation3D(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # --- Initialize PBR ---
        simplepbr.init(
            enable_shadows=True,
            shadow_bias=0.005
        )
        self.setBackgroundColor(0, 0, 0, 1)

        # --- Load models ---
        self.earth = loader.loadModel("models/earth/scene.gltf")
        self.starship = loader.loadModel("models/starship/starship.gltf")

        self.earth.reparentTo(self.render)
        self.starship.reparentTo(self.render)

        min_pt, max_pt = self.earth.getTightBounds()

        if min_pt is None:
            self.earth.flattenStrong()
            min_pt, max_pt = self.earth.getTightBounds()

        size = max_pt - min_pt
        dx, dy, dz = size.x, size.y, size.z
        target_diameter = min(dx, dy, dz)

        sx = target_diameter / dx
        sy = target_diameter / dy
        sz = target_diameter / dz

        self.earth.setScale(sx, sy, sz)

        diameter_x = size.x
        diameter_y = size.y
        diameter_z = size.z

        print("Earth model diameters:")
        print("X:", diameter_x)
        print("Y:", diameter_y)
        print("Z:", diameter_z)

        diameter_mean = (diameter_x + diameter_y + diameter_z) / 3
        print("Mean diameter:", diameter_mean)

        ground_diameter = min(diameter_x, diameter_y, diameter_z)

        scale_factor = (6_371_000.0+130_000) / (diameter_mean / 2)
        self.earth.setScale(scale_factor)

        # --- Starship sizing ---
        min_s, max_s = self.starship.getTightBounds()
        if min_s is None:
            self.starship.flattenStrong()
            min_s, max_s = self.starship.getTightBounds()

        size_s = max_s - min_s
        starship_height_model = size_s.z
        print("Starship model height:", starship_height_model)

        # Scale to real size
        TARGET_HEIGHT = 50.0  # meters
        scale_starship = TARGET_HEIGHT / starship_height_model
        self.starship.setScale(scale_starship)

        # Recompute bounds after scaling (world space)
        min_s, max_s = self.starship.getTightBounds(self.render)
        starship_bottom_offset = -min_s.z

        # Place on Earth's surface
        EARTH_RADIUS = 6_371_000.0
        self.starship.setPos(0, EARTH_RADIUS, 0)
        # self.starship.setPos(0, 0, EARTH_RADIUS)
        # self.starship.setPos(EARTH_RADIUS, 0, 0)

        # Disable default mouse camera
        # self.disableMouse()

        # Attach camera to Starship
        self.camera.reparentTo(self.starship)

        # Place camera next to Starship (meters)
        # self.camera.setPos(20, -40, 15)  # right, back, up
        self.camera.lookAt(0, 0, 25)  # look at mid-body

        # --- Lighting setup ---
        self.setup_lighting()

    def setup_lighting(self):
        # ======================
        # Key light (sun)
        # ======================
        key_light = DirectionalLight("key_light")
        key_light.setColor(Vec4(1.0, 0.95, 0.9, 1))  # warm sunlight
        key_light.setShadowCaster(True, 4096, 4096)

        key_np = self.render.attachNewNode(key_light)
        key_np.setHpr(45, -60, 0)  # angle down
        self.render.setLight(key_np)

        # ======================
        # Fill light
        # ======================
        fill_light = DirectionalLight("fill_light")
        fill_light.setColor(Vec4(0.4, 0.45, 0.5, 1))  # cool fill

        fill_np = self.render.attachNewNode(fill_light)
        fill_np.setHpr(-60, -30, 0)
        self.render.setLight(fill_np)

        # ======================
        # Rim light (edge highlight)
        # ======================
        rim_light = DirectionalLight("rim_light")
        rim_light.setColor(Vec4(0.6, 0.6, 0.7, 1))

        rim_np = self.render.attachNewNode(rim_light)
        rim_np.setHpr(180, -20, 0)
        self.render.setLight(rim_np)

        # ======================
        # Ambient (very low)
        # ======================
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.03, 0.03, 0.03, 1))

        amb_np = self.render.attachNewNode(ambient)
        self.render.setLight(amb_np)

app = Animation3D()
app.run()
