from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, loadPrcFileData
import simplepbr
from panda3d.core import DirectionalLight, AmbientLight, Vec4

# Optional: set window size for testing
loadPrcFileData("", "win-size 1024 768")

# Minimal LLA -> ECEF converter
def lla_to_ecef(lat_deg, lon_deg, alt_m, R=6_371_000.0):
    import numpy as np
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r = R + alt_m
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return Vec3(x, y, z)

class TestStarshipPosition(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        simplepbr.init(enable_shadows=False)
        self.setBackgroundColor(0, 0, 0, 1)

        # Load Earth model
        self.earth = loader.loadModel("models/earth/scene.gltf")
        self.earth.reparentTo(self.render)
        # ======================
        # Earth scaling
        # ======================
        min_pt, max_pt = self.earth.getTightBounds()
        if min_pt is None:
            self.earth.flattenStrong()
            min_pt, max_pt = self.earth.getTightBounds()

        size = max_pt - min_pt
        diameter_mean = (size.x + size.y + size.z) / 3

        scale_factor = (6_371_000.0 + 130_000) / (diameter_mean / 2)
        self.earth.setScale(scale_factor)

        # Load Starship model
        self.starship = loader.loadModel("models/starship/starship.gltf")
        self.starship.reparentTo(self.render)

        # Scale Starship roughly 50 meters tall
        min_s, max_s = self.starship.getTightBounds()
        height_model = (max_s - min_s).z
        self.starship.setScale(50 / height_model)

        # ============================
        # Set Starship at known coordinates
        # ============================
        # test_lat = 28.5721   # Kennedy Space Center, FL
        # test_lon = -80.6480
        # test_alt = 200_000    # 100 km altitude

        test_lat = 36.1408  # degrees North
        test_lon = -5.3536  # degrees East (West is negative)
        test_alt = 130_000  # 100 km

        pos = lla_to_ecef(test_lat, test_lon, test_alt)
        print("ECEF position:", pos)
        self.earth.setPos(-pos)

        # ============================
        # Camera
        # ============================


        # ============================
        # Lighting
        # ============================
        self.setup_lighting()

    def setup_lighting(self):
        key_light = DirectionalLight("key_light")
        key_light.setColor(Vec4(1, 1, 0.9, 1))
        key_np = self.render.attachNewNode(key_light)
        key_np.setHpr(45, -60, 0)
        self.render.setLight(key_np)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.2, 0.2, 0.2, 1))
        amb_np = self.render.attachNewNode(ambient)
        self.render.setLight(amb_np)


app = TestStarshipPosition()
app.run()
