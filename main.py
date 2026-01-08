import tkinter as tk
import random
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# =============================
#        参数（从这里改）
# =============================
CANVAS_SIZE = 720
RADIUS = 320  # 大圆半径
CENTER = CANVAS_SIZE // 2

# 数量
N_CELLS = 90
N_VIRUSES = 50
N_ANTIBODIES = 14
N_LEUKOCYTES = 6

# 尺寸
CELL_R_SMALL = 10
CELL_R_LARGE = 18
VIRUS_R = 7
AB_Y_SIZE = 7
AB_R_FOR_COLLISION = 3  # 抗体碰撞半径（用于细胞障碍物）
LEUKOCYTE_R = 10

# 动画与运动
FPS = 60
CA_INTERVAL = 0.12        # 每隔多少秒做一次CA决策（离散方向更新）
TURN_SMOOTH = 0.45        # 速度方向平滑系数（越小越丝滑）

VIRUS_SPEED = 70.0        # px/s
AB_SPEED = 140.0          # px/s
CELL_SPEED = 18.0         # px/s 细胞慢速随机漂移
LEUKOCYTE_SPEED = 95.0    # px/s 白细胞速度
VIRUS_ATTACHED_SPEED_FACTOR = 0.8
VIRUS_ATTACHED_SPEED_DECAY = 0.2
VIRUS_ATTACHED_AVOID_SCALE = 0.12
CELL_ATTACHED_SPEED_FACTOR = 0.7
CELL_ATTACHED_SPEED_DECAY = 0.15

# 行为参数
VIRUS_ATTRACT_CELL = 0.25  # 病毒趋向细胞程度
AB_SENSE_RADIUS = 120.0    # 抗体感知半径
AB_CHASE = 0.85            # 抗体追逐强度
CAPTURE_DIST = 12.0        # 抗体捕获距离
VIRUS_AVOID_LEUKOCYTE = 0.7  # 病毒远离白细胞基础强度
LEUKOCYTE_SENSE_RADIUS = 160.0
LEUKOCYTE_CHASE = 0.9
AB_SPAWN_MIN = 1
AB_SPAWN_MAX = 3

# 新增：感染与爆发参数
INFECTION_PADDING = 2                   # 病毒“贴到细胞”判定阈值补偿（可调）
VIRUS_REPLICATION_TIME = 3.0              # 病毒繁殖时间（秒）
BURST_VIRUS_COUNT_SMALL = 6               # 小细胞破裂出现的病毒数量
BURST_VIRUS_COUNT_LARGE = 18              # 大细胞破裂出现的病毒数量

# 新增：细胞分裂/成长参数
CELL_DIVIDE_TIME_MIN = 10.0               # 分裂最短时间（秒）
CELL_DIVIDE_TIME_MAX = 22.0               # 分裂最长时间（秒）
CELL_GROW_TIME = 18.0                     # 小细胞长成大细胞的时间（秒）

# 颜色
BG_COLOR = "white"
CELL_COLOR = "#4C78A8"
CELL_INFECTED_COLOR = "#B279A2"
CELL_INFECTED_BOUND_COLOR = "#A05195"
CELL_DEAD_COLOR = "#7F7F7F"

VIRUS_COLOR = "#F58518"
VIRUS_BOUND_COLOR = "#E45756"
AB_COLOR = "#54A24B"
AB_FLASH_COLOR = "#E45756"
LEUKOCYTE_COLOR = "#F2F2F2"
LEUKOCYTE_OUTLINE = "#333333"
# =============================


# ---------- 数据结构 ----------
@dataclass
class Cell:
    x: float
    y: float
    vx: float
    vy: float
    r: float = CELL_R_LARGE
    state: str = "healthy"      # healthy | infected | dead
    burst_timer: float = 0.0    # infected -> countdown to burst
    grow_timer: float = CELL_GROW_TIME
    divide_timer: Optional[float] = None
    antibody_attached: int = 0


@dataclass
class Virus:
    x: float
    y: float
    vx: float
    vy: float
    attached: int = 0


@dataclass
class Antibody:
    x: float
    y: float
    vx: float
    vy: float
    flash: int = 0  # 捕获后闪烁若干帧


@dataclass
class Leukocyte:
    x: float
    y: float
    vx: float
    vy: float


# ---------- 工具 ----------
def rand_point_in_circle(r: float, margin: float = 0) -> Tuple[float, float]:
    a = random.random() * 2 * math.pi
    rr = math.sqrt(random.random()) * (r - margin)
    x = CENTER + rr * math.cos(a)
    y = CENTER + rr * math.sin(a)
    return x, y


def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def unit_vec(dx: float, dy: float) -> Tuple[float, float]:
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return 0.0, 0.0
    return dx / d, dy / d


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def burst_count_for_cell(cell: Cell) -> int:
    denom = CELL_R_LARGE - CELL_R_SMALL
    ratio = 1.0 if denom <= 0 else clamp((cell.r - CELL_R_SMALL) / denom, 0.0, 1.0)
    return int(round(lerp(BURST_VIRUS_COUNT_SMALL, BURST_VIRUS_COUNT_LARGE, ratio)))


def pick_discrete_direction(dx: float, dy: float, directions: List[Tuple[float, float]]) -> Tuple[float, float]:
    ux, uy = unit_vec(dx, dy)
    best = directions[0]
    best_dot = -1e9
    for vx, vy in directions:
        dot = ux * vx + uy * vy
        if dot > best_dot:
            best_dot = dot
            best = (vx, vy)
    return best


def reflect_off_circle(x: float, y: float, vx: float, vy: float, margin: float) -> Tuple[float, float, float, float]:
    dx = x - CENTER
    dy = y - CENTER
    d = math.hypot(dx, dy)
    limit = RADIUS - margin
    if d <= limit or d < 1e-9:
        return x, y, vx, vy

    nx, ny = dx / d, dy / d
    x = CENTER + nx * limit
    y = CENTER + ny * limit

    dot = vx * nx + vy * ny
    vx = vx - 2 * dot * nx
    vy = vy - 2 * dot * ny
    return x, y, vx, vy


def push_out_of_cells(x: float, y: float, vx: float, vy: float, r_obj: float, cells: List[Cell]) -> Tuple[float, float, float, float]:
    # healthy/infected 细胞作为障碍物；dead 不再阻挡（你也可以改成仍阻挡）
    for c in cells:
        if c.state == "dead":
            continue
        dx = x - c.x
        dy = y - c.y
        d = math.hypot(dx, dy)
        min_d = c.r + r_obj
        if d < min_d and d > 1e-9:
            nx, ny = dx / d, dy / d
            x = c.x + nx * min_d
            y = c.y + ny * min_d
            dot = vx * nx + vy * ny
            vx = vx - 1.8 * dot * nx
            vy = vy - 1.8 * dot * ny
        elif d < 1e-9:
            a = random.random() * 2 * math.pi
            x = c.x + math.cos(a) * min_d
            y = c.y + math.sin(a) * min_d
    return x, y, vx, vy


def push_out_of_other_cells(cell: Cell, cells: List[Cell]) -> None:
    for other in cells:
        if other is cell or other.state == "dead":
            continue
        dx = cell.x - other.x
        dy = cell.y - other.y
        d = math.hypot(dx, dy)
        min_d = cell.r + other.r
        if d < min_d and d > 1e-9:
            nx, ny = dx / d, dy / d
            overlap = min_d - d
            cell.x += nx * overlap * 0.6
            cell.y += ny * overlap * 0.6
            cell.vx -= nx * overlap * 0.4
            cell.vy -= ny * overlap * 0.4
        elif d < 1e-9:
            ang = random.random() * 2 * math.pi
            cell.x = other.x + math.cos(ang) * min_d
            cell.y = other.y + math.sin(ang) * min_d


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("丝滑 CA：抗体附着 + 白细胞清理 + 细胞感染爆发（圆形边界）")
        root.minsize(760, 820)

        self.top = tk.Frame(root)
        self.top.pack(side="top", fill="both", expand=True)
        self.bottom = tk.Frame(root)
        self.bottom.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(self.top, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=BG_COLOR)
        self.canvas.pack(padx=10, pady=10)

        self.btn = tk.Button(self.bottom, text="Start", width=12, command=self.toggle)
        self.btn.pack(side="left", padx=8, pady=8)

        self.btn_step = tk.Button(self.bottom, text="Step CA", width=12, command=self.step_ca_once)
        self.btn_step.pack(side="left", padx=8, pady=8)

        self.btn_reset = tk.Button(self.bottom, text="Reset", width=12, command=self.reset)
        self.btn_reset.pack(side="left", padx=8, pady=8)

        self.speed_scale = tk.Scale(self.bottom, from_=20, to=90, orient="horizontal",
                                    label="FPS", length=220)
        self.speed_scale.set(FPS)
        self.speed_scale.pack(side="right", padx=10)

        self.running = False
        self.after_id: Optional[str] = None

        # 统计
        self.captured = 0
        self.tick = 0
        self.infected_count = 0
        self.burst_count = 0
        self.elapsed_time = 0.0
        self.history: List[Tuple[float, int, int, int, int]] = []

        # 离散方向（16方向）
        self.directions = []
        for k in range(16):
            ang = 2 * math.pi * k / 16
            self.directions.append((math.cos(ang), math.sin(ang)))

        self.cells: List[Cell] = []
        self.viruses: List[Virus] = []
        self.antibodies: List[Antibody] = []
        self.leukocytes: List[Leukocyte] = []

        self.ca_accum = 0.0

        self.draw_static()
        self.reset()

    def draw_static(self):
        self.canvas.delete("static")
        r = RADIUS
        self.canvas.create_oval(CENTER - r, CENTER - r, CENTER + r, CENTER + r,
                                outline="#333", width=3, fill="#f8f8ff", tags=("static",))
        self.canvas.create_text(12, 12, anchor="nw",
                                text="CA决策(离散方向) + 连续运动(丝滑) + 抗体附着 + 白细胞清理",
                                fill="#444", font=("Helvetica", 12), tags=("static",))

    def reset(self):
        self.running = False
        self.btn.configure(text="Start")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.captured = 0
        self.tick = 0
        self.infected_count = 0
        self.burst_count = 0
        self.elapsed_time = 0.0
        self.history = []
        self.ca_accum = 0.0

        self.cells = []
        self.viruses = []
        self.antibodies = []
        self.leukocytes = []

        # 生成细胞（尽量不重叠）
        attempts = 0
        while len(self.cells) < N_CELLS and attempts < 6000:
            attempts += 1
            x, y = rand_point_in_circle(RADIUS, margin=70)
            ok = True
            for c in self.cells:
                if math.hypot(x - c.x, y - c.y) < (CELL_R_LARGE * 2 + 14):
                    ok = False
                    break
            if ok:
                ang = random.random() * 2 * math.pi
                vx = CELL_SPEED * math.cos(ang)
                vy = CELL_SPEED * math.sin(ang)
                self.cells.append(Cell(x=x, y=y, vx=vx, vy=vy, r=CELL_R_LARGE,
                                       grow_timer=CELL_GROW_TIME,
                                       divide_timer=random.uniform(CELL_DIVIDE_TIME_MIN, CELL_DIVIDE_TIME_MAX)))

        # 生成病毒
        while len(self.viruses) < N_VIRUSES:
            x, y = rand_point_in_circle(RADIUS, margin=20)
            if any(math.hypot(x - c.x, y - c.y) < (c.r + VIRUS_R + 2) for c in self.cells):
                continue
            ang = random.random() * 2 * math.pi
            vx = VIRUS_SPEED * math.cos(ang)
            vy = VIRUS_SPEED * math.sin(ang)
            self.viruses.append(Virus(x=x, y=y, vx=vx, vy=vy))

        # 生成抗体
        while len(self.antibodies) < N_ANTIBODIES:
            x, y = rand_point_in_circle(RADIUS, margin=15)
            if any(math.hypot(x - c.x, y - c.y) < (c.r + AB_R_FOR_COLLISION + 2) for c in self.cells):
                continue
            ang = random.random() * 2 * math.pi
            vx = AB_SPEED * math.cos(ang)
            vy = AB_SPEED * math.sin(ang)
            self.antibodies.append(Antibody(x=x, y=y, vx=vx, vy=vy))

        # 生成白细胞
        while len(self.leukocytes) < N_LEUKOCYTES:
            x, y = rand_point_in_circle(RADIUS, margin=25)
            if any(math.hypot(x - c.x, y - c.y) < (c.r + LEUKOCYTE_R + 4) for c in self.cells):
                continue
            ang = random.random() * 2 * math.pi
            vx = LEUKOCYTE_SPEED * math.cos(ang)
            vy = LEUKOCYTE_SPEED * math.sin(ang)
            self.leukocytes.append(Leukocyte(x=x, y=y, vx=vx, vy=vy))

        self.record_history()
        self.render()

    def toggle(self):
        self.running = not self.running
        self.btn.configure(text="Pause" if self.running else "Start")
        if self.running:
            self.loop()
        else:
            self.show_timeline_chart()

    def step_ca_once(self):
        self.ca_step()
        self.render()

    def loop(self):
        if not self.running:
            return
        fps = max(10, int(self.speed_scale.get()))
        dt = 1.0 / fps

        self.animate_step(dt)
        self.render()
        if not self.running:
            return
        self.after_id = self.root.after(int(1000 / fps), self.loop)

    # ---------- 连续动画步 ----------
    def animate_step(self, dt: float):
        self.tick += 1
        self.elapsed_time += dt
        self.ca_accum += dt

        # CA决策步
        if self.ca_accum >= CA_INTERVAL:
            self.ca_accum %= CA_INTERVAL
            self.ca_step()

        # 连续移动：细胞
        for c in self.cells:
            if c.state == "dead":
                continue
            speed_factor = 1.0
            if c.antibody_attached > 0:
                speed_factor = max(0.3, CELL_ATTACHED_SPEED_FACTOR - c.antibody_attached * CELL_ATTACHED_SPEED_DECAY)
            c.x += c.vx * dt
            c.y += c.vy * dt
            c.vx *= speed_factor
            c.vy *= speed_factor
            c.x, c.y, c.vx, c.vy = reflect_off_circle(c.x, c.y, c.vx, c.vy, margin=c.r)
            push_out_of_other_cells(c, self.cells)

        # 细胞成长与分裂
        self.cell_growth_and_division(dt)

        # 连续移动：病毒
        for v in self.viruses:
            v.x += v.vx * dt
            v.y += v.vy * dt
            v.x, v.y, v.vx, v.vy = reflect_off_circle(v.x, v.y, v.vx, v.vy, margin=VIRUS_R)
            v.x, v.y, v.vx, v.vy = push_out_of_cells(v.x, v.y, v.vx, v.vy, VIRUS_R, self.cells)

        # 连续移动：抗体
        for a in self.antibodies:
            if a.flash > 0:
                a.flash -= 1
            a.x += a.vx * dt
            a.y += a.vy * dt
            a.x, a.y, a.vx, a.vy = reflect_off_circle(a.x, a.y, a.vx, a.vy, margin=AB_R_FOR_COLLISION)
            a.x, a.y, a.vx, a.vy = push_out_of_cells(a.x, a.y, a.vx, a.vy, AB_R_FOR_COLLISION, self.cells)

        # 连续移动：白细胞
        for w in self.leukocytes:
            w.x += w.vx * dt
            w.y += w.vy * dt
            w.x, w.y, w.vx, w.vy = reflect_off_circle(w.x, w.y, w.vx, w.vy, margin=LEUKOCYTE_R)
            w.x, w.y, w.vx, w.vy = push_out_of_cells(w.x, w.y, w.vx, w.vy, LEUKOCYTE_R, self.cells)

        # 新增：感染逻辑（病毒贴到细胞 → 细胞变色并开始倒计时 → 爆发）
        self.infection_step(dt)

        # 抗体附着
        self.capture_check()

        # 白细胞清理
        self.leukocyte_cleanup()
        self.record_history()

    def record_history(self):
        live_cells = sum(1 for c in self.cells if c.state != "dead")
        self.history.append((self.elapsed_time, len(self.leukocytes), live_cells, len(self.viruses), len(self.antibodies)))

    def show_timeline_chart(self):
        if not self.history:
            return
        chart = tk.Toplevel(self.root)
        chart.title("数量时间线")
        width, height = 760, 420
        margin_left, margin_right = 60, 20
        margin_top, margin_bottom = 40, 50
        canvas = tk.Canvas(chart, width=width, height=height, bg="white")
        canvas.pack(fill="both", expand=True)

        times = [h[0] for h in self.history]
        leukocytes = [h[1] for h in self.history]
        cells = [h[2] for h in self.history]
        viruses = [h[3] for h in self.history]
        antibodies = [h[4] for h in self.history]

        max_time = max(times) if times else 1.0
        max_count = max(max(leukocytes), max(cells), max(viruses), max(antibodies), 1)

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        x0 = margin_left
        y0 = height - margin_bottom

        canvas.create_line(x0, y0, x0 + plot_w, y0, fill="#333")
        canvas.create_line(x0, y0, x0, y0 - plot_h, fill="#333")
        canvas.create_text(x0, y0 + 25, text="时间 (s)", anchor="nw", fill="#333")
        canvas.create_text(10, margin_top - 10, text="数量", anchor="nw", fill="#333")

        def to_xy(t: float, value: int) -> Tuple[float, float]:
            x = x0 + (t / max_time) * plot_w
            y = y0 - (value / max_count) * plot_h
            return x, y

        def draw_series(values: List[int], color: str, label: str, y_offset: int):
            points = []
            for t, v in zip(times, values):
                points.extend(to_xy(t, v))
            if len(points) >= 4:
                canvas.create_line(points, fill=color, width=2)
            legend_x = x0 + plot_w - 120
            legend_y = margin_top + y_offset
            canvas.create_line(legend_x, legend_y + 6, legend_x + 18, legend_y + 6, fill=color, width=3)
            canvas.create_text(legend_x + 26, legend_y, text=label, anchor="nw", fill="#333")

        draw_series(leukocytes, LEUKOCYTE_OUTLINE, "白细胞", 0)
        draw_series(cells, CELL_COLOR, "普通细胞", 20)
        draw_series(viruses, VIRUS_COLOR, "病毒", 40)
        draw_series(antibodies, AB_COLOR, "抗体", 60)

    # ---------- CA决策步：只更新“速度方向” ----------
    def ca_step(self):
        # 细胞：慢速、无目的乱动
        for c in self.cells:
            if c.state == "dead":
                continue
            ang = random.random() * 2 * math.pi
            tx, ty = math.cos(ang), math.sin(ang)
            ddx, ddy = pick_discrete_direction(tx, ty, self.directions)
            nvx, nvy = ddx * CELL_SPEED, ddy * CELL_SPEED
            c.vx = (1.0 - TURN_SMOOTH) * c.vx + TURN_SMOOTH * nvx
            c.vy = (1.0 - TURN_SMOOTH) * c.vy + TURN_SMOOTH * nvy

        # 病毒：随机游走 + 轻微向最近“未死亡细胞”靠近 + 远离白细胞
        live_cells = [c for c in self.cells if c.state != "dead"]
        for v in self.viruses:
            ang = random.random() * 2 * math.pi
            rx, ry = math.cos(ang), math.sin(ang)

            if live_cells and v.attached <= 0:
                nearest = min(live_cells, key=lambda c: dist2(v.x, v.y, c.x, c.y))
                cx, cy = nearest.x - v.x, nearest.y - v.y
                cux, cuy = unit_vec(cx, cy)
            else:
                cux, cuy = 0.0, 0.0

            if self.leukocytes:
                nearest_w = min(self.leukocytes, key=lambda w: dist2(v.x, v.y, w.x, w.y))
                wx, wy = v.x - nearest_w.x, v.y - nearest_w.y
                wux, wuy = unit_vec(wx, wy)
            else:
                wux, wuy = 0.0, 0.0

            avoid_scale = 1.0 + v.attached * VIRUS_ATTACHED_AVOID_SCALE
            avoid_strength = VIRUS_AVOID_LEUKOCYTE * avoid_scale
            base = max(0.0, 1.0 - VIRUS_ATTRACT_CELL - avoid_strength)
            tx = base * rx + VIRUS_ATTRACT_CELL * cux + avoid_strength * wux
            ty = base * ry + VIRUS_ATTRACT_CELL * cuy + avoid_strength * wuy

            ddx, ddy = pick_discrete_direction(tx, ty, self.directions)
            speed_factor = max(0.2, VIRUS_ATTACHED_SPEED_FACTOR - v.attached * VIRUS_ATTACHED_SPEED_DECAY)
            speed = VIRUS_SPEED * speed_factor
            nvx, nvy = ddx * speed, ddy * speed
            v.vx = (1.0 - TURN_SMOOTH) * v.vx + TURN_SMOOTH * nvx
            v.vy = (1.0 - TURN_SMOOTH) * v.vy + TURN_SMOOTH * nvy

        # 抗体：感知半径内找最近未附着病毒，否则随机
        sense2 = AB_SENSE_RADIUS * AB_SENSE_RADIUS
        for a in self.antibodies:
            target: Optional[Virus] = None
            best_d2 = sense2
            for v in self.viruses:
                if v.attached:
                    continue
                d2 = dist2(a.x, a.y, v.x, v.y)
                if d2 < best_d2:
                    best_d2 = d2
                    target = v

            if target is None:
                ang = random.random() * 2 * math.pi
                tx, ty = math.cos(ang), math.sin(ang)
            else:
                dx, dy = target.x - a.x, target.y - a.y
                tux, tuy = unit_vec(dx, dy)
                ang = random.random() * 2 * math.pi
                rx, ry = math.cos(ang), math.sin(ang)
                tx = AB_CHASE * tux + (1.0 - AB_CHASE) * rx
                ty = AB_CHASE * tuy + (1.0 - AB_CHASE) * ry

            ddx, ddy = pick_discrete_direction(tx, ty, self.directions)
            nvx, nvy = ddx * AB_SPEED, ddy * AB_SPEED
            a.vx = (1.0 - TURN_SMOOTH) * a.vx + TURN_SMOOTH * nvx
            a.vy = (1.0 - TURN_SMOOTH) * a.vy + TURN_SMOOTH * nvy

        # 白细胞：追踪被标记目标（附着病毒/感染细胞/死亡细胞）
        sense2 = LEUKOCYTE_SENSE_RADIUS * LEUKOCYTE_SENSE_RADIUS
        targets: List[Tuple[float, float]] = []
        targets.extend((v.x, v.y) for v in self.viruses if v.attached > 0)
        targets.extend((c.x, c.y) for c in self.cells if c.state in ("infected", "dead") or c.antibody_attached > 0)
        for w in self.leukocytes:
            target_pos: Optional[Tuple[float, float]] = None
            best_d2 = sense2
            for tx_pos, ty_pos in targets:
                d2 = dist2(w.x, w.y, tx_pos, ty_pos)
                if d2 < best_d2:
                    best_d2 = d2
                    target_pos = (tx_pos, ty_pos)
            if target_pos is None:
                ang = random.random() * 2 * math.pi
                tx, ty = math.cos(ang), math.sin(ang)
            else:
                dx, dy = target_pos[0] - w.x, target_pos[1] - w.y
                tux, tuy = unit_vec(dx, dy)
                ang = random.random() * 2 * math.pi
                rx, ry = math.cos(ang), math.sin(ang)
                tx = LEUKOCYTE_CHASE * tux + (1.0 - LEUKOCYTE_CHASE) * rx
                ty = LEUKOCYTE_CHASE * tuy + (1.0 - LEUKOCYTE_CHASE) * ry
            ddx, ddy = pick_discrete_direction(tx, ty, self.directions)
            nvx, nvy = ddx * LEUKOCYTE_SPEED, ddy * LEUKOCYTE_SPEED
            w.vx = (1.0 - TURN_SMOOTH) * w.vx + TURN_SMOOTH * nvx
            w.vy = (1.0 - TURN_SMOOTH) * w.vy + TURN_SMOOTH * nvy

    def cell_growth_and_division(self, dt: float):
        if not self.cells:
            return
        updated_cells = []
        newborn_cells: List[Cell] = []

        for c in self.cells:
            if c.state == "dead":
                updated_cells.append(c)
                continue

            if CELL_GROW_TIME <= 0:
                c.grow_timer = CELL_GROW_TIME
                c.r = CELL_R_LARGE
                if c.divide_timer is None:
                    c.divide_timer = random.uniform(CELL_DIVIDE_TIME_MIN, CELL_DIVIDE_TIME_MAX)
            elif c.grow_timer < CELL_GROW_TIME:
                c.grow_timer = min(CELL_GROW_TIME, c.grow_timer + dt)
                progress = c.grow_timer / CELL_GROW_TIME
                c.r = lerp(CELL_R_SMALL, CELL_R_LARGE, progress)
                if c.r >= CELL_R_LARGE - 1e-3 and c.divide_timer is None:
                    c.divide_timer = random.uniform(CELL_DIVIDE_TIME_MIN, CELL_DIVIDE_TIME_MAX)

            if c.state == "healthy" and c.r >= CELL_R_LARGE - 1e-3 and c.divide_timer is not None:
                c.divide_timer -= dt
                if c.divide_timer <= 0:
                    newborn_cells.extend(self.divide_cell(c))
                    continue

            updated_cells.append(c)

        if newborn_cells:
            updated_cells.extend(newborn_cells)
            for newborn in newborn_cells:
                push_out_of_other_cells(newborn, updated_cells)
        self.cells = updated_cells

    def divide_cell(self, cell: Cell) -> List[Cell]:
        ang = random.random() * 2 * math.pi
        offset = max(CELL_R_SMALL + 2, cell.r * 0.6)
        dx = math.cos(ang) * offset
        dy = math.sin(ang) * offset
        positions = [(cell.x + dx, cell.y + dy), (cell.x - dx, cell.y - dy)]
        children = []
        for x, y in positions:
            if not self._inside_big_circle(x, y, margin=CELL_R_SMALL):
                dx_c, dy_c = x - CENTER, y - CENTER
                d = math.hypot(dx_c, dy_c) or 1.0
                nx, ny = dx_c / d, dy_c / d
                limit = RADIUS - CELL_R_SMALL
                x = CENTER + nx * limit
                y = CENTER + ny * limit
            ang_v = random.random() * 2 * math.pi
            vx = CELL_SPEED * math.cos(ang_v)
            vy = CELL_SPEED * math.sin(ang_v)
            children.append(Cell(x=x, y=y, vx=vx, vy=vy, r=CELL_R_SMALL, grow_timer=0.0))
        return children

    # ---------- 新增：感染/爆发 ----------
    def infection_step(self, dt: float):
        if not self.cells or not self.viruses:
            return

        # 1) 病毒贴到健康细胞 -> 感染（细胞变色）+ 该病毒“进入细胞”（删除）
        new_viruses = []
        removed_by_infection = 0

        # 为了效率：先把细胞分组（这里只做简单遍历，规模不大够用）
        for v in self.viruses:
            if v.attached > 0:
                new_viruses.append(v)
                continue
            infected = False
            for c in self.cells:
                if c.state != "healthy":
                    continue
                infection_dist = c.r + VIRUS_R + INFECTION_PADDING
                if dist2(v.x, v.y, c.x, c.y) <= infection_dist * infection_dist:
                    # 感染发生
                    c.state = "infected"
                    c.burst_timer = VIRUS_REPLICATION_TIME
                    self.infected_count += 1
                    infected = True
                    removed_by_infection += 1
                    break
            if not infected:
                new_viruses.append(v)

        if removed_by_infection:
            self.viruses = new_viruses

        # 2) 感染细胞倒计时 -> 破裂爆发
        for c in self.cells:
            if c.state == "infected":
                c.burst_timer -= dt
                if c.burst_timer <= 0:
                    self.burst_count += 1
                    c.state = "dead"
                    c.antibody_attached = 0

                    burst_count = burst_count_for_cell(c)

                    # 爆发产生病毒：从细胞附近喷出
                    for _ in range(burst_count):
                        ang = random.random() * 2 * math.pi
                        # 出生点：细胞边缘附近稍微外移一点
                        rr = c.r + VIRUS_R + random.random() * 6.0
                        x = c.x + rr * math.cos(ang)
                        y = c.y + rr * math.sin(ang)

                        # 保证在大圆内；如果不在就往内拉一点
                        if not self._inside_big_circle(x, y, margin=VIRUS_R):
                            # 把点拉回到大圆内
                            dx, dy = x - CENTER, y - CENTER
                            d = math.hypot(dx, dy) or 1.0
                            nx, ny = dx / d, dy / d
                            limit = RADIUS - VIRUS_R
                            x = CENTER + nx * limit
                            y = CENTER + ny * limit

                        # 速度：随机方向，略带“喷射”效果（速度有抖动）
                        sp = VIRUS_SPEED * (0.9 + random.random() * 0.5)
                        vx = sp * math.cos(ang)
                        vy = sp * math.sin(ang)

                        self.viruses.append(Virus(x=x, y=y, vx=vx, vy=vy))

    def _inside_big_circle(self, x: float, y: float, margin: float = 0) -> bool:
        dx = x - CENTER
        dy = y - CENTER
        return (dx * dx + dy * dy) <= (RADIUS - margin) ** 2

    # ---------- 抗体捕获 ----------
    def capture_check(self):
        if not self.viruses and not self.cells:
            return
        cap2 = CAPTURE_DIST * CAPTURE_DIST
        remaining_antibodies = []

        for a in self.antibodies:
            attached = False
            for v in self.viruses:
                if dist2(v.x, v.y, a.x, a.y) <= cap2:
                    v.attached += 1
                    a.flash = 8
                    self.captured += 1
                    attached = True
                    break
            if attached:
                continue
            for c in self.cells:
                if c.state != "infected":
                    continue
                if dist2(c.x, c.y, a.x, a.y) <= cap2:
                    c.antibody_attached += 1
                    a.flash = 8
                    attached = True
                    break
            if not attached:
                remaining_antibodies.append(a)

        self.antibodies = remaining_antibodies

    def _spawn_antibodies(self, x: float, y: float, count: int) -> None:
        if count <= 0:
            return
        for _ in range(count):
            ang = random.random() * 2 * math.pi
            rr = 6 + random.random() * 8
            px = x + rr * math.cos(ang)
            py = y + rr * math.sin(ang)
            if not self._inside_big_circle(px, py, margin=AB_R_FOR_COLLISION):
                dx, dy = px - CENTER, py - CENTER
                d = math.hypot(dx, dy) or 1.0
                nx, ny = dx / d, dy / d
                limit = RADIUS - AB_R_FOR_COLLISION
                px = CENTER + nx * limit
                py = CENTER + ny * limit
            ang_v = random.random() * 2 * math.pi
            vx = AB_SPEED * math.cos(ang_v)
            vy = AB_SPEED * math.sin(ang_v)
            self.antibodies.append(Antibody(x=px, y=py, vx=vx, vy=vy))

    def leukocyte_cleanup(self):
        if not self.leukocytes:
            return
        virus_removed = set()
        cell_removed = set()
        virus_dist2 = (LEUKOCYTE_R + VIRUS_R) ** 2

        for w in self.leukocytes:
            for idx, v in enumerate(self.viruses):
                if idx in virus_removed:
                    continue
                if dist2(w.x, w.y, v.x, v.y) <= virus_dist2:
                    virus_removed.add(idx)
                    spawn_count = random.randint(AB_SPAWN_MIN, AB_SPAWN_MAX)
                    self._spawn_antibodies(v.x, v.y, spawn_count)

            for idx, c in enumerate(self.cells):
                if idx in cell_removed:
                    continue
                if c.state == "dead" or c.state == "infected":
                    if dist2(w.x, w.y, c.x, c.y) <= (LEUKOCYTE_R + c.r) ** 2:
                        cell_removed.add(idx)
                        spawn_count = random.randint(AB_SPAWN_MIN, AB_SPAWN_MAX)
                        self._spawn_antibodies(c.x, c.y, spawn_count)

        if virus_removed:
            self.viruses = [v for idx, v in enumerate(self.viruses) if idx not in virus_removed]
        if cell_removed:
            self.cells = [c for idx, c in enumerate(self.cells) if idx not in cell_removed]

    # ---------- 绘制 ----------
    def render(self):
        self.canvas.delete("dyn")

        # 细胞（按状态变色）
        for c in self.cells:
            if c.state == "healthy":
                col = CELL_COLOR
            elif c.state == "infected":
                col = CELL_INFECTED_BOUND_COLOR if c.antibody_attached > 0 else CELL_INFECTED_COLOR
            else:
                col = CELL_DEAD_COLOR

            self.canvas.create_oval(c.x - c.r, c.y - c.r, c.x + c.r, c.y + c.r,
                                    fill=col, outline="", tags=("dyn",))
            # 核心点
            self.canvas.create_oval(c.x - 3, c.y - 3, c.x + 3, c.y + 3,
                                    fill="#2F4B7C", outline="", tags=("dyn",))

            # 感染倒计时显示（可选）
            if c.state == "infected":
                self.canvas.create_text(c.x, c.y - c.r - 10, text=f"{max(0.0, c.burst_timer):.1f}s",
                                        fill="#333", font=("Helvetica", 10), tags=("dyn",))

        # 病毒
        for v in self.viruses:
            v_col = VIRUS_BOUND_COLOR if v.attached > 0 else VIRUS_COLOR
            self.canvas.create_oval(v.x - VIRUS_R, v.y - VIRUS_R, v.x + VIRUS_R, v.y + VIRUS_R,
                                    fill=v_col, outline="", tags=("dyn",))

        # 抗体（Y）
        for a in self.antibodies:
            col = AB_FLASH_COLOR if a.flash > 0 else AB_COLOR
            s = AB_Y_SIZE
            x, y = a.x, a.y
            self.canvas.create_line(x, y, x - s, y - s, fill=col, width=2, tags=("dyn",))
            self.canvas.create_line(x, y, x + s, y - s, fill=col, width=2, tags=("dyn",))
            self.canvas.create_line(x, y, x, y + s + 2, fill=col, width=2, tags=("dyn",))

        # 白细胞
        for w in self.leukocytes:
            self.canvas.create_oval(w.x - LEUKOCYTE_R, w.y - LEUKOCYTE_R,
                                    w.x + LEUKOCYTE_R, w.y + LEUKOCYTE_R,
                                    fill=LEUKOCYTE_COLOR, outline=LEUKOCYTE_OUTLINE, width=2, tags=("dyn",))

        # HUD
        self.canvas.create_text(12, 42, anchor="nw",
                                text=(f"Tick:{self.tick}  Viruses:{len(self.viruses)}  "
                                      f"Antibodies:{len(self.antibodies)}  Captured:{self.captured}  "
                                      f"Infected:{self.infected_count}  Bursts:{self.burst_count}  "
                                      f"Leukocytes:{len(self.leukocytes)}"),
                                fill="#111", font=("Helvetica", 12), tags=("dyn",))


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
