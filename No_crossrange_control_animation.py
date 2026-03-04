import reentripy as rpy
import numpy as np
from pystdatm import density

import math

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties
from panda3d.core import load_prc_file
import simplepbr
from panda3d.core import DirectionalLight, AmbientLight, Vec3, Vec4

from panda3d.core import Mat3, Quat
from panda3d.core import TransformState


from direct.gui.DirectGui import (
    DirectButton, DirectWaitBar, DirectFrame, DirectLabel
)
from direct.task import Task

from panda3d.core import TextNode

from panda3d.core import load_prc_file_data

load_prc_file_data("", """
fullscreen true
win-size 1920 1080
undecorated true
""")


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
sc.load_remaining_range_map("starship_apqc_remaining_range.npz")

# ------------------------------
# Orbit definition: Conditions for IFT test flights ( more or less )
# ------------------------------
apogee = 213_000.0
perigee = -15_000.0
altitude = 200_000.0   # current altitude

inclination = np.deg2rad(26.8)     # Starship-like
arg_perigee = np.deg2rad(146)
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

import os

cache_file = "reentry_run_starship_IFT_s_turns.npz"

# cache_file = "reentry_run_starship_IFT_no_s_turns.npz"


if os.path.exists(cache_file):
    print("Loading cached reentry run...")
    data = np.load(cache_file)
    times = data["times"]
    altitudes = data["altitudes"]
    speeds = data["speeds"]
    machs = data["machs"]
    bank_angles = data["bank_angles"]
    g_forces = data["g_forces"]
    descent_rates = data["descent_rates"]
    positions = data["positions"]
    lon = data["lon"]
    lat = data["lat"]
    heat_loads = data["heat_loads"]
    heat_fluxes = data["heat_fluxes"]
    aoas = data["aoas"]
    sogs_vecs = data["sogs_vecs"]
else:
    print("Running reentry simulation...")
    times, altitudes, speeds, machs, bank_angles, g_forces, descent_rates, positions, lon, lat, heat_loads, heat_fluxes, aoas, sogs_vecs = sc.run_reentry(
        gif=False,
        controller="aPQC",
        DTLH=True
    )

    np.savez(cache_file,
        times=times,
        altitudes=altitudes,
        speeds=speeds,
        machs=machs,
        bank_angles=bank_angles,
        g_forces=g_forces,
        descent_rates=descent_rates,
        positions=positions,
        lon=lon,
        lat=lat,
        heat_loads=heat_loads,
        heat_fluxes=heat_fluxes,
        aoas=aoas,
        sogs_vecs=sogs_vecs
    )


from panda3d.core import LineSegs, NodePath


def rolling_average(data, window=5):
    """
    Applies a simple moving average filter to a 1D array.
    Args:
        data: 1D numpy array
        window: number of samples to average over
    Returns:
        Smoothed 1D numpy array
    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def smooth_signal(data, window=5):
    smoothed = rolling_average(data, window)
    pad = np.full(window-1, smoothed[0])  # pad start with first value
    return np.concatenate([pad, smoothed])

bank_angles = smooth_signal(bank_angles, window=100)
aoas = smooth_signal(aoas, window=100)


def altitude_to_sky_color(alt_m):
    """
    Returns an RGB tuple for the sky color as a function of altitude.
    alt_m: altitude in meters
    """
    # Altitudes in meters
    min_alt = 0       # sea level
    max_alt = 100_000 # 100 km

    # Clamp altitude
    alt_m = max(min_alt, min(alt_m, max_alt))

    # Define space color (black) and daylight blue
    space_color = np.array([0.0, 0.0, 0.0])        # near black
    daylight_color = np.array([0.53, 0.81, 0.92])   # light sky blue

    t = (alt_m / max_alt) ** 0.5  # faster transition near ground
    color = space_color * t + daylight_color * (1 - t)

    return tuple(color)


def lla_to_ecef(lat, lon, alt_m, R=6_371_000.0):
    # print(np.rad2deg(lon), np.rad2deg(lat))
    lon = lon+np.deg2rad(197.85)
    if lon>180:
        lon-=360

    r = R + alt_m

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return Vec3(x, y, z)


traj_lat = lat
traj_lon = lon
traj_alt = altitudes
traj_mach = machs
traj_time = times

def descent_angle(vel: Vec3, up_local: Vec3) -> float:
    """
    Compute descent angle (deg) relative to vertical.
    vel: velocity vector (Panda3D Vec3)
    up_local: vector from Earth's center to craft (normalized)
    Returns:
        descent angle in degrees (0 = horizontal, 90 = straight down)
    """
    vel_norm = vel.normalized()
    up_norm = up_local.normalized()

    # Angle between -up_local and velocity
    cos_gamma = (-up_norm).dot(vel_norm)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)  # numerical safety
    gamma_rad = np.arccos(cos_gamma)
    gamma_deg = 90-np.degrees(gamma_rad)
    return gamma_deg


from panda3d.core import LineSegs, NodePath, Vec3
from panda3d.core import (
    GeomVertexFormat, GeomVertexData, Geom, GeomVertexWriter,
    GeomLinestrips, GeomNode, Vec4, NodePath, LineSegs, TextNode
)

class FastSciFiGraph:
    """
    v6 — Full sci-fi graph system with:
    - Grid
    - Axes
    - Background
    - X and Y labels
    - Floating dynamic Y-value label (T1 position)
    - GPU-accelerated static geometry
    """
    def __init__(
        self, x, y,
        pos=(0.67, 0, 0.48),
        size=(0.34, 0.26),
        color_past=(1,1,1,1),
        color_future=(0.25,0.25,0.25,1),
        axis_color=(0.7,0.7,0.7,1),
        grid_color=(0.3,0.3,0.3,0.4),
        bgcolor=(0,0,0,0.35),
        xlabel="Time (s)",
        ylabel="Value",
        label="Name",
        yunits=""
    ):
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label
        self.yunits = yunits

        self.pos, self.size = pos, size
        self.color_past = Vec4(*color_past)
        self.color_future = Vec4(*color_future)

        self.N = len(x)
        self.xmin, self.xmax = float(x.min()), float(x.max())
        self.ymin, self.ymax = float(y.min()), float(y.max())

        self.root = NodePath("graph_root")
        self.root.reparentTo(base.render2d)
        self.root.setTransparency(True)

        px, py, pz = pos
        w, h = size

        # ---------------- BACKGROUND ----------------
        bg = LineSegs()
        bg.setColor(*bgcolor)
        bg.moveTo(px, py, pz)
        bg.drawTo(px+w, py, pz)
        bg.drawTo(px+w, py, pz+h)
        bg.drawTo(px, py, pz+h)
        bg.drawTo(px, py, pz)
        self.bg_np = NodePath(bg.create())
        self.bg_np.reparentTo(self.root)

        # ---------------- GRID ----------------
        grid = LineSegs()
        grid.setColor(*grid_color)
        grid.setThickness(1)

        NX = 8
        NY = 5

        for i in range(1, NX):
            xx = px + (i/NX)*w
            grid.moveTo(xx, py, pz)
            grid.drawTo(xx, py, pz+h)

        for j in range(1, NY):
            zz = pz + (j/NY)*h
            grid.moveTo(px, py, zz)
            grid.drawTo(px+w, py, zz)

        self.grid_np = NodePath(grid.create())
        self.grid_np.reparentTo(self.root)

        # ---------------- AXES ----------------
        ax = LineSegs()
        ax.setColor(*axis_color)
        ax.setThickness(2)

        ax.moveTo(px,   py, pz)
        ax.drawTo(px+w, py, pz)

        ax.moveTo(px, py, pz)
        ax.drawTo(px, py, pz+h)

        self.axis_np = NodePath(ax.create())
        self.axis_np.reparentTo(self.root)

        # ---------------- AXIS LABELS ----------------
        font = TextNode.getDefaultFont()

        xlbl = TextNode("xlabel")
        xlbl.setText(self.xlabel)
        xlbl.setFont(font)
        xlbl.setTextColor(1,1,1,0.9)
        xlbl.setAlign(TextNode.ACenter)
        xnp = self.root.attachNewNode(xlbl)
        xnp.setScale(0.035)
        xnp.setPos(px + w*0.5, py, pz - 0.04)

        ylbl = TextNode("ylabel")
        ylbl.setText(self.ylabel)
        ylbl.setFont(font)
        ylbl.setTextColor(1,1,1,0.9)
        ylbl.setAlign(TextNode.ACenter)
        ynp = self.root.attachNewNode(ylbl)
        ynp.setScale(0.035)
        ynp.setHpr(0, 0, 90)
        ynp.setPos(px - 0.045, py, pz + h*0.5)

        lbl = TextNode("label")
        lbl.setText(self.label)
        lbl.setFont(font)
        lbl.setTextColor(1, 1, 1, 0.9)
        lbl.setAlign(TextNode.ACenter)
        lnp = self.root.attachNewNode(lbl)
        lnp.setScale(0.02)
        lnp.setPos(px + w * 0.5, py, pz + h + 0.01)

        # ---------------- DYNAMIC FLOATING VALUE LABEL ----------------
        self.vlabel = TextNode("vlabel")
        self.vlabel.setFont(font)
        self.vlabel.setTextColor(1,1,1,0.9)
        self.vlabel.setAlign(TextNode.ARight)
        self.vlabel_np = self.root.attachNewNode(self.vlabel)
        self.vlabel_np.setScale(0.02)

        # ---------------- GPU LINE STRIP ----------------
        fmt = GeomVertexFormat.getV3c4()
        self.vdata = GeomVertexData("graph", fmt, Geom.UH_dynamic)
        self.vdata.setNumRows(self.N)

        vwriter = GeomVertexWriter(self.vdata, "vertex")
        cwriter = GeomVertexWriter(self.vdata, "color")

        for i in range(self.N):
            xn = (self.x[i]-self.xmin)/(self.xmax-self.xmin)
            yn = (self.y[i]-self.ymin)/(self.ymax-self.ymin)

            vx = px + xn*w
            vy = py
            vz = pz + yn*h

            vwriter.addData3(vx,vy,vz)
            cwriter.addData4(*self.color_future)

        prim = GeomLinestrips(Geom.UH_static)
        for i in range(self.N):
            prim.addVertex(i)
        prim.closePrimitive()

        geom = Geom(self.vdata)
        geom.addPrimitive(prim)
        node = GeomNode("graph_line")
        node.addGeom(geom)

        self.line_np = NodePath(node)
        self.line_np.reparentTo(self.root)

    # ------------ FAST UPDATE PER FRAME ------------
    def update(self, t):

        # find current index
        idx = np.searchsorted(self.x, t)
        if idx >= self.N:
            idx = self.N - 1

        current_value = self.y[idx]
        norm_y = (current_value - self.ymin)/(self.ymax-self.ymin)

        px, py, pz = self.pos
        w, h = self.size

        # update floating value text
        self.vlabel.setText(
            f"{current_value:.2f}{self.yunits}"
        )

        self.vlabel_np.setPos(
            px - 0.01, py,
            pz + norm_y*h
        )

        # update color buffer
        cwriter = GeomVertexWriter(self.vdata, "color")
        for i in range(self.N):
            if self.x[i] <= t:
                cwriter.setData4f(*self.color_past)
            else:
                cwriter.setData4f(*self.color_future)

    def set_visible(self, flag):
        if flag: self.root.show()
        else:    self.root.hide()


def draw_vectors(origin: Vec3, vel: Vec3, right: Vec3, up: Vec3, parent):
    """
    Draw velocity, right, and up vectors as colored arrows.
    origin: Vec3 start point
    vel, right, up: normalized Vec3 directions
    parent: NodePath to attach to (e.g., self.render)
    """
    scale = 50  # adjust length for visibility

    ls = LineSegs()
    ls.setThickness(3)

    # Velocity (red)
    ls.setColor(1, 0, 0, 1)
    ls.moveTo(origin)
    ls.drawTo(origin + vel * scale)

    # Right (green)
    ls.setColor(0, 1, 0, 1)
    ls.moveTo(origin)
    ls.drawTo(origin + right * scale)

    # Up (blue)
    ls.setColor(0, 0, 1, 1)
    ls.moveTo(origin)
    ls.drawTo(origin + up * scale)

    np = NodePath(ls.create())
    np.reparentTo(parent)
    return np


class Animation3D(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # --- Initialize PBR ---
        simplepbr.init(enable_shadows=False)
        self.setBackgroundColor(0, 0, 0, 1)

        self.extra_scale = 1.0

        # ======================
        # Load models
        # ======================
        self.earth = loader.loadModel("models/earth/scene.gltf")
        self.starship = loader.loadModel("models/starship/starship.gltf")

        # --- Root for Earth motion ---
        self.earth_root = self.render.attachNewNode("earth_root")
        self.earth.reparentTo(self.earth_root)

        # Starship stays in world space
        self.starship.reparentTo(self.render)

        # ======================
        # Earth scaling
        # ======================
        min_pt, max_pt = self.earth.getTightBounds()
        if min_pt is None:
            self.earth.flattenStrong()
            min_pt, max_pt = self.earth.getTightBounds()

        size = max_pt - min_pt
        diameter_mean = (size.x + size.y + size.z) / 3

        scale_factor = (6_371_000.0 + 111_000) / (diameter_mean / 2)
        self.earth.setScale(scale_factor * self.extra_scale)

        # ======================
        # Starship scaling
        # ======================
        min_s, max_s = self.starship.getTightBounds()
        if min_s is None:
            self.starship.flattenStrong()
            min_s, max_s = self.starship.getTightBounds()

        starship_height_model = (max_s - min_s).z
        TARGET_HEIGHT = 50.0  # meters
        self.starship.setScale(
            (TARGET_HEIGHT / starship_height_model) * self.extra_scale
        )

        # ======================
        # Starship fixed position
        # ======================
        self.starship.setPos(0, 0, 0)

        # ======================
        # Camera
        # ======================
        self.camera.reparentTo(self.starship)
        self.camera.setPos(20, -40, 15)
        self.camera.lookAt(0, 0, 25)

        # ======================
        # Lighting
        # ======================
        self.setup_lighting()

        # ======================
        # Trajectory (attached to Earth)
        # ======================
        self.draw_trajectory(lat, lon, altitudes)

        # ======================
        # Animation state
        # ======================
        self.traj_index = 0
        self.playing = False
        self.time_scale = 1  # x1, x2, x5, x10

        # ======================
        # UI
        # ======================
        self.create_ui()

        # ======================
        # Animation task
        # ======================
        self.taskMgr.add(self.update_animation, "update_animation")

        lens = self.cam.node().getLens()
        lens.setNear(1.0)
        lens.setFar(5.0e6)  # 1 million meters

        self.camera.reparentTo(self.render)

    # =====================================================
    # Lighting
    # =====================================================
    def setup_lighting(self):
        key_light = DirectionalLight("key_light")
        key_light.setColor(Vec4(0.5, 0.5, 0.5, 1))
        key_np = self.starship.attachNewNode(key_light)
        key_np.setHpr(-45, -60, 0)
        self.render.setLight(key_np)

        fill_light = DirectionalLight("fill_light")
        fill_light.setColor(Vec4(0.4, 0.45, 0.5, 1))
        fill_np = self.starship.attachNewNode(fill_light)
        fill_np.setHpr(60, -30, 0)
        self.render.setLight(fill_np)

        rim_light = DirectionalLight("rim_light")
        rim_light.setColor(Vec4(0.6, 0.6, 0.7, 1))
        rim_np = self.render.attachNewNode(rim_light)
        rim_np.setHpr(-180, -20, 0)
        self.render.setLight(rim_np)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.1, 0.1, 0.1, 1))
        amb_np = self.starship.attachNewNode(ambient)
        self.render.setLight(amb_np)

    # =====================================================
    # Trajectory drawing (Earth space)
    # =====================================================
    def draw_trajectory(self, lat, lon, alt):
        segs = LineSegs()
        segs.setThickness(3.0)  # thicker line
        segs.setColor(0.5, 0.5, 0.5, 1.0)  # fully opaque white

        first = True
        for la, lo, al in zip(lat, lon, alt):
            p = lla_to_ecef(la, lo, al)
            if first:
                segs.moveTo(p)
                first = False
            else:
                segs.drawTo(p)

        traj_np = NodePath(segs.create())
        traj_np.reparentTo(self.earth_root)

    # =====================================================
    # UI
    # =====================================================
    def create_ui(self):
        aspect = base.getAspectRatio()

        # ===== Unified color theme =====
        BG_DARK = (0, 0, 0, 0.38)
        BG_LIGHT = (0.15, 0.15, 0.15, 0.6)
        FG_WHITE = (1, 1, 1, 1)
        FG_GREY = (0.85, 0.85, 0.88, 1)
        ACCENT = (0.2, 0.75, 1.0, 1.0)

        # Better font rendering
        font = loader.loadFont("models/fonts/Roboto-Medium.ttf")
        font.setPixelsPerUnit(60)

        # ===== Bottom HUD container =====
        self.ui_frame = DirectFrame(
            frameColor=BG_DARK,
            frameSize=(-aspect, aspect, -0.14, 0.13),
            pos=(0, 0, -0.9)
        )

        # Slight shadow under the HUD
        shadow = DirectFrame(
            frameColor=(0, 0, 0, 0.2),
            frameSize=(-aspect, aspect, -0.01, 0.01),
            pos=(0, 0, -0.9)
        )
        shadow.reparentTo(self.ui_frame)

        # ===== PLAY / PAUSE BUTTON =====
        self.play_button = DirectButton(
            text="Start",
            scale=0.06,
            pos=(-aspect + 0.05+0.15, 0, 0.015),
            frameColor=BG_LIGHT,
            frameSize=(-2.5, 2.5, -0.7, 1.3),
            text_fg=FG_WHITE,
            text_font=font,
            # relief="ridge",
            command=self.toggle_play,
            parent=self.ui_frame
        )

        # ===== CENTRAL T‑Clock =====
        self.t_clock = DirectLabel(
            text="T+00:00:00",
            scale=0.08,
            pos=(0, 0, 0.02),
            text_align=TextNode.ACenter,
            frameColor=(0.0, 0.0, 0.0, 0.0),
            text_fg=FG_WHITE,
            text_shadow=(0, 0, 0, 0.9),
            text_font=font,
            parent=self.ui_frame
        )

        # ===== TELEMETRY PANEL (Right side) =====
        self.telemetry = DirectLabel(
            text="",
            scale=0.045,
            pos=(aspect - 0.22, 0, 0.02),
            text_align=TextNode.ARight,
            text_fg=FG_GREY,
            text_shadow=(0, 0, 0, 1),
            text_font=font,
            parent=self.ui_frame
        )

        # ===== PROGRESS BAR (Full width) =====
        self.progress = DirectWaitBar(
            range=len(traj_time),
            value=0,
            frameSize=(-aspect, aspect, -0.02, 0.02),
            pos=(0, 0, -0.1),
            parent=self.ui_frame,
            barColor=ACCENT,
            frameColor=(0, 0, 0, 0.3)
        )

        # ===== TIME WARP BUTTONS (same height as Start, narrower) =====

        btn_scale = 0.06

        # Start button frameSize height: (-0.7 → 1.3)  ==> keep this
        # Narrower width: reduce from (-2.5 → 2.5) to (-0.65 → 0.65)
        warp_frame = (-1.0, 1.0, -0.7, 1.3)

        start_btn_x = -aspect + 0.05 + 0.06
        base_y = 0.015

        # Horizontal spacing
        spacing = 0.12

        warp_buttons = [
            ("x1", 1),
            ("x2", 2),
            ("x5", 5),
            ("x10", 10),
        ]

        for i, (label, value) in enumerate(warp_buttons):
            DirectButton(
                text=label,
                scale=btn_scale,
                pos=(start_btn_x + 0.32 + i * spacing, 0, base_y),
                frameColor=BG_LIGHT,
                frameSize=warp_frame,  # ← same height as Start, narrower width
                text_fg=FG_WHITE,
                text_font=font,
                # relief="flat",
                command=self.set_time_scale,
                extraArgs=[value],
                parent=self.ui_frame
            )
        self.graphs_visible = True
        # ===== TOGGLE GRAPHS BUTTON =====
        DirectButton(
            text="GRAPHS",
            scale=0.06,
            pos=(start_btn_x + 0.32 + 4 * spacing + 0.22, 0, base_y),
            frameColor=BG_LIGHT,
            frameSize=(-2.5, 2.5, -0.7, 1.3),  # same height as Start/Pause
            text_fg=FG_WHITE,
            text_font=font,
            # relief="flat",
            command=self.toggle_graphs,
            parent=self.ui_frame
        )
        # ====== Create Graphs ======

        GRAPH_SIZE = (0.38, 0.30)  # width, height
        GRAPH_CENTER = -0.19  # center reference (usually 0)
        GRAPH_HSPACE = 0.65  # horizontal offset from center
        GRAPH_POS_Y = 0.6  # Y position of top graphs
        GRAPH_VSPACE = 0.4  # vertical spacing between rows

        # ---------- RIGHT COLUMN (4 graphs): ALT → MACH → HEAT FLUX → DESCENT RATE ----------

        # RIGHT TOP — ALTITUDE (red)
        self.alt_graph = FastSciFiGraph(
            traj_time, traj_alt / 1000,
            pos=(GRAPH_CENTER + GRAPH_HSPACE, 0, GRAPH_POS_Y),
            size=GRAPH_SIZE,
            color_past=(1, 0.2, 0.2, 1),
            color_future=(0.2, 0.05, 0.05, 1),
            xlabel="", ylabel="", label="Altitude", yunits=" km"
        )

        # RIGHT MID — MACH (green)
        self.mach_graph = FastSciFiGraph(
            traj_time, traj_mach,
            pos=(GRAPH_CENTER + GRAPH_HSPACE, 0, GRAPH_POS_Y - GRAPH_VSPACE),
            size=GRAPH_SIZE,
            color_past=(0.2, 1, 0.2, 1),
            color_future=(0.05, 0.2, 0.05, 1),
            xlabel="", ylabel="", label="Mach", yunits=""
        )

        # RIGHT LOWER-MID — HEAT FLUX (purple)  <<< swapped to RIGHT column
        self.heat_graph = FastSciFiGraph(
            traj_time, heat_fluxes / 1000.0,  # kW/cm^2 if original is W/cm^2
            pos=(GRAPH_CENTER + GRAPH_HSPACE, 0, GRAPH_POS_Y - 2 * GRAPH_VSPACE),
            size=GRAPH_SIZE,
            color_past=(0.9, 0.2, 1.0, 1),  # Purple
            color_future=(0.25, 0.1, 0.3, 1),
            xlabel="", ylabel="", label="Heat Flux", yunits=" kW/cm2"
        )

        # RIGHT BOTTOM — DESCENT RATE (white)
        self.descent_graph = FastSciFiGraph(
            traj_time, descent_rates,  # assumes m/s
            pos=(GRAPH_CENTER + GRAPH_HSPACE, 0, GRAPH_POS_Y - 3 * GRAPH_VSPACE),
            size=GRAPH_SIZE,
            color_past=(1.0, 1.0, 1.0, 1),  # White past
            color_future=(0.35, 0.35, 0.35, 1),  # Grey future
            xlabel="", ylabel="", label="Descent Rate", yunits=" m/s"
        )

        # ---------- LEFT COLUMN (3 graphs): G → BANK → AOA (swapped from right) ----------

        # LEFT TOP — G‑FORCE (orange)
        self.g_graph = FastSciFiGraph(
            traj_time, g_forces,
            pos=(GRAPH_CENTER - GRAPH_HSPACE, 0, GRAPH_POS_Y),
            size=GRAPH_SIZE,
            color_past=(1.0, 0.6, 0.1, 1),  # Orange
            color_future=(0.3, 0.15, 0.05, 1),
            xlabel="", ylabel="", label="G-Force", yunits=" g"
        )

        # LEFT MID — BANK ANGLE (blue)
        self.bank_graph = FastSciFiGraph(
            traj_time, bank_angles,
            pos=(GRAPH_CENTER - GRAPH_HSPACE, 0, GRAPH_POS_Y - GRAPH_VSPACE),
            size=GRAPH_SIZE,
            color_past=(0.2, 0.55, 1.0, 1),  # Blue
            color_future=(0.07, 0.18, 0.3, 1),
            xlabel="", ylabel="", label="Bank Angle", yunits="°"
        )

        # LEFT BOTTOM — AOA (cyan)  <<< swapped to LEFT column
        self.aoa_graph = FastSciFiGraph(
            traj_time, aoas,
            pos=(GRAPH_CENTER - GRAPH_HSPACE, 0, GRAPH_POS_Y - 2 * GRAPH_VSPACE),
            size=GRAPH_SIZE,
            color_past=(0.1, 0.9, 1.0, 1),  # Cyan
            color_future=(0.05, 0.35, 0.40, 1),
            xlabel="", ylabel="", label="AOA", yunits="°"
        )


    def set_time_scale(self, scale):
        self.time_scale = scale

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button["text"] = " Pause " if self.playing else " Start "

    # =====================================================
    # Animation loop (Earth moves!)
    # =====================================================
    def update_animation(self, task):

        alt_m = traj_alt[self.traj_index]  # current altitude
        self.setBackgroundColor(*altitude_to_sky_color(alt_m), 1)

        if not self.playing:
            return Task.cont

        if self.traj_index >= len(traj_time):
            self.playing = False
            self.play_button["text"] = "Start"
            return Task.done

        # --- Earth moves opposite to spacecraft ---
        pos = lla_to_ecef(
            traj_lat[self.traj_index],
            traj_lon[self.traj_index],
            traj_alt[self.traj_index]
        )
        self.earth_root.setPos(-pos)


        # --- Telemetry ---
        t_remaining = traj_time[-1] - traj_time[self.traj_index]

        hours = int(t_remaining // 3600)
        minutes = int((t_remaining % 3600) // 60)
        seconds = int(t_remaining % 60)

        self.t_clock["text"] = f"T-{hours:02d}:{minutes:02d}:{seconds:02d}"

        alt_km = traj_alt[self.traj_index] / 1000
        mach = traj_mach[self.traj_index]

        self.telemetry["text"] = (
            f"Alt: {alt_km:6.1f} km\n"
            f"Mach: {mach:5.2f}"
        )

        self.progress["value"] = self.traj_index
        self.traj_index += self.time_scale
        self.traj_index = min(self.traj_index, len(traj_time) - 1)

        # --- Starship orientation using SOG vector ---

        # 1. Velocity vector (unit)
        starship_pos = lla_to_ecef(traj_lat[self.traj_index],
                                   traj_lon[self.traj_index],
                                   traj_alt[self.traj_index])
        starship_pos_last = lla_to_ecef(traj_lat[self.traj_index-1],
                                   traj_lon[self.traj_index-1],
                                   traj_alt[self.traj_index-1])
        vel_np = Vec3(*(starship_pos-starship_pos_last)).normalized()
        vel_np = vel_np / np.linalg.norm(vel_np)
        forward = Vec3(*vel_np)

        # 2. Local up vector (Earth center to Starship)

        up = starship_pos.normalized()

        # Try to compute right; fallback if too small
        right_candidate = up.cross(forward)
        if right_candidate.length() < 1e-1:
            # Forward is nearly vertical; keep previous right if available
            try:
                right = self.prev_right  # store last frame's right
            except AttributeError:
                # first frame fallback: pick any perpendicular vector
                right = Vec3(1, 0, 0)
        else:
            right = right_candidate.normalized()
            self.prev_right = right  # store for next frame

        # Recompute up to ensure orthogonality
        up = forward.cross(right).normalized()

        # --- 1. Forward, right, up vectors ---
        forward = vel_np.normalized()  # along velocity
        up = up.normalized()  # local up
        right = right.normalized()  # perpendicular to forward and up

        # --- 2. Apply bank (rotate right/up around forward) ---
        bank_rad = -math.radians(bank_angles[self.traj_index])
        cos_b = math.cos(bank_rad)
        sin_b = math.sin(bank_rad)

        right_banked = right * cos_b + up * sin_b
        up_banked = -right * sin_b + up * cos_b

        # --- 3. Apply AoA (rotate forward/up around right_banked) ---
        aoa_rad = math.radians(aoas[self.traj_index])
        cos_a = math.cos(aoa_rad)
        sin_a = math.sin(aoa_rad)

        forward_final = forward * cos_a + up_banked * sin_a
        up_final = -forward * sin_a + up_banked * cos_a

        # --- 4. Build rotation matrix / final orientation ---
        mat = Mat3()
        mat.setRow(0, right_banked)  # X-axis
        mat.setRow(1, up_final)  # Y-axis
        mat.setRow(2, forward_final)  # Z-axis

        final_quat = Quat()
        final_quat.setFromMatrix(mat)

        self.starship.setQuat(final_quat)

        # ===== Update Graphs =====
        if self.graphs_visible:
            current_t = traj_time[self.traj_index]
            self.alt_graph.update(current_t)
            self.mach_graph.update(current_t)
            self.bank_graph.update(current_t)
            self.g_graph.update(current_t)
            self.aoa_graph.update(current_t)
            self.heat_graph.update(current_t)
            self.descent_graph.update(current_t)



        # 2. Draw them vectors (attach to render or starship root)
        # draw_vectors(starship_pos, vel_np, right, up, self.earth_root)
        # Desired camera offset relative to Starship's velocity frame
        cam_back = 300.0  # behind the Starship along velocity vector
        cam_up = 50.0  # above Starship

        # Camera in world space: Starship position minus forward vector * back offset
        camera_pos = Vec3(0,0,0) - forward * cam_back \
                     + up * cam_up \
                     + 0.0

        self.camera.setPos(camera_pos)
        self.camera.lookAt(Vec3(0,0,0), up)

        return Task.cont


    def toggle_graphs(self):
        self.graphs_visible = not self.graphs_visible
        self.alt_graph.set_visible(self.graphs_visible)
        self.mach_graph.set_visible(self.graphs_visible)
        self.g_graph.set_visible(self.graphs_visible)
        self.bank_graph.set_visible(self.graphs_visible)
        self.aoa_graph.set_visible(self.graphs_visible)
        self.heat_graph.set_visible(self.graphs_visible)
        self.descent_graph.set_visible(self.graphs_visible)



app = Animation3D()
app.run()
