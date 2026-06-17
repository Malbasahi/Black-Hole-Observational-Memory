# physics_blackhole_gpu.py
# pip install pygame moderngl numpy
# python physics_blackhole_gpu.py

import sys
from dataclasses import dataclass

import moderngl
import numpy as np
import pygame


WIDTH, HEIGHT = 1536, 864
FPS = 60


VERTEX_SHADER = """
#version 330

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330

uniform vec2  u_resolution;
uniform float u_time;

uniform float u_mass;
uniform float u_spin;
uniform float u_accretion;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_distance;

uniform float u_show_disk;
uniform float u_show_ring;
uniform float u_show_grid;
uniform float u_show_stars;

in vec2 v_uv;
out vec4 fragColor;

#define PI 3.14159265359

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;

    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p *= 2.02;
        a *= 0.5;
    }

    return v;
}

vec3 hotColor(float x) {
    x = clamp(x, 0.0, 1.0);

    vec3 black = vec3(0.025, 0.010, 0.003);
    vec3 red   = vec3(0.70, 0.20, 0.035);
    vec3 gold  = vec3(1.00, 0.56, 0.16);
    vec3 white = vec3(1.00, 0.90, 0.68);

    vec3 c = mix(black, red, smoothstep(0.00, 0.34, x));
    c = mix(c, gold, smoothstep(0.30, 0.72, x));
    c = mix(c, white, smoothstep(0.66, 1.00, x));

    return c;
}

vec3 tonemap(vec3 c) {
    c = vec3(1.0) - exp(-c * 1.18);
    c = pow(c, vec3(0.84));
    return c;
}

vec3 cameraPosition() {
    return vec3(
        sin(u_yaw) * cos(u_pitch),
        sin(u_pitch),
        cos(u_yaw) * cos(u_pitch)
    ) * u_distance;
}

mat3 cameraBasis(vec3 ro) {
    vec3 forward = normalize(-ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), forward));
    vec3 up = normalize(cross(forward, right));
    return mat3(right, up, forward);
}

vec3 getRay(vec2 p, vec3 ro) {
    mat3 cam = cameraBasis(ro);
    return normalize(cam * vec3(p, 1.48));
}

vec3 starfield(vec3 rd) {
    if (u_show_stars < 0.5) return vec3(0.0);

    vec3 d = normalize(rd);

    vec2 uv = vec2(
        atan(d.z, d.x) / (2.0 * PI) + 0.5,
        asin(d.y) / PI + 0.5
    );

    vec3 col = vec3(0.0);

    // Milky-way-like dust band
    float band = exp(-pow(abs(d.y) * 3.6, 2.0));
    float neb1 = fbm(uv * vec2(7.0, 3.0) + vec2(0.15, 0.23));
    float neb2 = fbm(uv * vec2(18.0, 7.0) + vec2(-0.4, 0.1));

    vec3 dust =
        vec3(0.055, 0.065, 0.090) * neb1 * band * 0.75 +
        vec3(0.120, 0.075, 0.035) * neb2 * band * 0.35;

    col += dust;

    // Multi-scale star layers
    for (int layer = 0; layer < 3; layer++) {
        float scale = layer == 0 ? 900.0 : layer == 1 ? 1500.0 : 2600.0;
        float threshold = layer == 0 ? 0.9960 : layer == 1 ? 0.9980 : 0.9991;

        vec2 p = uv * vec2(scale, scale * 0.52);
        vec2 id = floor(p);
        vec2 f = fract(p);

        float rnd = hash21(id + float(layer) * 19.17);
        float star = smoothstep(threshold, 1.0, rnd);

        float dist = length(f - 0.5);
        float core = exp(-dist * 42.0);
        float glow = exp(-dist * 14.0) * 0.28;

        float temp = hash21(id + 8.3);
        vec3 starCol = mix(
            vec3(0.62, 0.75, 1.00),
            vec3(1.00, 0.86, 0.58),
            temp
        );

        float brightness = mix(0.35, 1.9, hash21(id + 4.7));

        col += star * starCol * brightness * (core + glow);
    }

    // Rare bright stars
    vec2 p = uv * vec2(420.0, 220.0);
    vec2 id = floor(p);
    vec2 f = fract(p);

    float rnd = hash21(id + 91.7);
    float brightStar = smoothstep(0.9988, 1.0, rnd);
    float dist = length(f - 0.5);

    vec3 brightCol = mix(
        vec3(0.7, 0.82, 1.0),
        vec3(1.0, 0.78, 0.45),
        hash21(id + 11.0)
    );

    col += brightStar * brightCol * exp(-dist * 8.0) * 1.8;
    col += brightStar * brightCol * exp(-dist * 25.0) * 2.4;

    return col;
}

float gridLine(float x, float w) {
    float g = abs(fract(x) - 0.5);
    return 1.0 - smoothstep(w * 0.45, w, g);
}

vec3 sampleGrid(vec3 p) {
    if (u_show_grid < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float r = length(p.xz) + 0.0001;

    if (r < horizon * 1.05) return vec3(0.0);

    float well = -0.82 * (horizon * 2.2) / (r + horizon * 1.22);
    float d = abs(p.y - well);

    float sheet = exp(-d * 45.0);

    float pull = 0.35 * horizon * horizon / (r * r + horizon * horizon);
    vec2 q = p.xz * (1.0 - pull);

    float spacing = 0.22;
    float gx = gridLine(q.x / spacing, 0.075);
    float gz = gridLine(q.y / spacing, 0.075);

    float line = max(gx, gz);

    float fade = smoothstep(5.6, 1.2, r);
    float holeFade = smoothstep(horizon * 1.10, horizon * 2.15, r);

    return vec3(0.045, 0.150, 0.235) * line * sheet * fade * holeFade * 0.55;
}

vec3 sampleDisk(vec3 p, vec3 rd) {
    if (u_show_disk < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float inner = 0.74;
    float outer = 3.35;

    if (length(p) < horizon * 1.03) return vec3(0.0);

    // A real tilted disk basis in world space.
    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(p, diskX);
    float y = dot(p, diskNormal);
    float z = dot(p, diskZ);

    float r = length(vec2(x, z));
    float a = atan(z, x);

    if (r < inner || r > outer) return vec3(0.0);

    float thickness = 0.085 + 0.060 * exp(-1.05 * (r - inner));
    float vertical = exp(-(y * y) / (2.0 * thickness * thickness));

    float radial = exp(-1.18 * max(r - inner, 0.0));

    float spiral =
        0.68
        + 0.18 * fbm(vec2(r * 7.0 - u_time * 0.22, a * 2.8))
        + 0.09 * sin(24.0 * r + 5.5 * a - u_time * 1.45)
        + 0.04 * sin(82.0 * r - u_time * 0.8);

    vec3 orbitalDir = normalize(-sin(a) * diskX + cos(a) * diskZ);

    float doppler = 1.0 + 1.70 * u_spin * dot(orbitalDir, -rd);
    doppler = clamp(doppler, 0.18, 3.05);

    float heat = exp(-0.70 * max(r - inner, 0.0));

    // Add extra brightness close to photon orbit, but in 3D space, not screen space.
    float photonBoost = 1.0 + 0.42 * exp(-pow(r - 0.95, 2.0) / 0.035);

    float brightness = vertical * radial * spiral * doppler * heat * photonBoost * u_accretion;
    brightness = pow(max(brightness, 0.0), 0.64);

    return hotColor(brightness) * brightness;
}

void main() {
    vec2 uv = v_uv;
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / u_resolution.y;

    vec3 ro = cameraPosition();
    vec3 rd = getRay(p, ro);

    float horizon = 0.62;

    vec3 rayPos = ro;
    vec3 rayDir = rd;

    vec3 emission = vec3(0.0);
    float opacity = 0.0;

    bool absorbed = false;

    float stepSize = 0.045;

    // Real bent-ray integration.
    // This is the main upgrade: the ray bends around the black hole,
    // so the disk and shadow change when the camera orbits.
    for (int i = 0; i < 180; i++) {
        float r = length(rayPos);

        if (r < horizon) {
            absorbed = true;
            break;
        }

        vec3 diskCol = sampleDisk(rayPos, rayDir);
        vec3 gridCol = sampleGrid(rayPos);

        float localPower = length(diskCol);

        emission += diskCol * 0.055 * (1.0 - opacity);
        opacity += localPower * 0.012 * (1.0 - opacity);

        emission += gridCol * 0.040 * (1.0 - opacity);

        // Schwarzschild-inspired bending toward the mass.
        vec3 towardBH = normalize(-rayPos);
        float bend = 0.030 * (u_mass / 10.0) / (r * r + 0.20);

        // Stronger near photon region.
        bend += 0.020 * exp(-pow(r - 0.92, 2.0) / 0.12);

        rayDir = normalize(rayDir + towardBH * bend * stepSize);

        rayPos += rayDir * stepSize;

        if (length(rayPos) > 9.0 || opacity > 0.96) break;
    }

    vec3 col = emission;

    // If not absorbed by the event horizon, show lensed star background.
    if (!absorbed) {
        col += starfield(rayDir) * (1.0 - opacity) * 1.35;
    }

    // Horizon edge glow emerges from bent nearby disk light, not a fixed screen circle.
    float closestImpact = length(cross(ro, rd));
    if (u_show_ring > 0.5) {
        float rim = exp(-pow(closestImpact - horizon * 1.10, 2.0) / 0.006);
        float outer = exp(-pow(closestImpact - horizon * 1.55, 2.0) / 0.065);

        col += vec3(1.0, 0.66, 0.28) * rim * 0.18 * u_accretion;
        col += vec3(1.0, 0.38, 0.10) * outer * 0.030 * u_accretion;
    }

    // Dark gravitational bowl near the event horizon.
    float bowl = exp(-pow(closestImpact - horizon * 1.25, 2.0) / 0.045);
    col *= 1.0 - bowl * 0.30;

    // If absorbed, keep only foreground disk/grid light already collected.
    // This makes the hole move as a real absorbing sphere.
    if (absorbed) {
        col *= 0.28;
    }

    float vignette = smoothstep(1.60, 0.12, length(p));
    col *= mix(0.34, 1.0, vignette);

    col = tonemap(col);

    fragColor = vec4(col, 1.0);
}
"""


@dataclass
class State:
    mass: float = 9.87
    spin: float = 0.85
    accretion: float = 0.80
    time_scale: float = 1.0

    yaw: float = 0.0
    pitch: float = 0.08
    distance: float = 4.2

    show_disk: bool = True
    show_ring: bool = True
    show_grid: bool = True
    show_stars: bool = True
    show_ui: bool = True

    dragging: bool = False
    dragging_slider: str | None = None


def clamp(x, a, b):
    return max(a, min(b, x))


def draw_text(screen, font, txt, x, y, color=(235, 235, 235)):
    screen.blit(font.render(txt, True, color), (x, y))


def draw_panel(screen, rect):
    surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    surf.fill((12, 12, 16, 182))
    pygame.draw.rect(surf, (60, 60, 70, 150), surf.get_rect(), 1, border_radius=8)
    screen.blit(surf, rect[:2])


def draw_checkbox(screen, font, label, checked, x, y):
    pygame.draw.rect(screen, (220, 220, 225), (x, y, 16, 16), 2, border_radius=3)

    if checked:
        pygame.draw.line(screen, (240, 240, 240), (x + 3, y + 8), (x + 7, y + 13), 2)
        pygame.draw.line(screen, (240, 240, 240), (x + 7, y + 13), (x + 14, y + 3), 2)

    draw_text(screen, font, label, x + 24, y - 1)


def draw_slider(screen, font, label, value, x, y, minv, maxv):
    draw_text(screen, font, label, x, y)
    draw_text(screen, font, f"{value:.2f}", x + 150, y)

    bar_x = x
    bar_y = y + 30
    bar_w = 170

    pygame.draw.rect(screen, (55, 55, 60), (bar_x, bar_y, bar_w, 6), border_radius=3)

    f = (value - minv) / (maxv - minv)
    knob_x = int(bar_x + f * bar_w)

    pygame.draw.circle(screen, (220, 220, 225), (knob_x, bar_y + 3), 7)


def draw_button(screen, font, label, rect):
    pygame.draw.rect(screen, (45, 45, 50), rect, border_radius=6)
    pygame.draw.rect(screen, (70, 70, 78), rect, 1, border_radius=6)

    tx = rect[0] + rect[2] // 2 - font.size(label)[0] // 2
    ty = rect[1] + 7
    draw_text(screen, font, label, tx, ty)


def draw_ui(screen, font, state, fps):
    if not state.show_ui:
        return

    muted = (170, 175, 185)

    draw_panel(screen, (14, 50, 220, 180))
    draw_text(screen, font, "Simulation Info", 26, 68)
    draw_text(screen, font, f"BH Mass: {state.mass:.2f} M*", 26, 100, muted)
    draw_text(screen, font, f"Spin (a): {state.spin:.2f}", 26, 126, muted)
    draw_text(screen, font, f"Accretion Rate: {state.accretion:.2f}", 26, 152, muted)
    draw_text(screen, font, f"Time Scale: {state.time_scale:.2f}x", 26, 178, muted)
    draw_text(screen, font, f"Frame Rate: {fps:.0f} FPS", 26, 204, muted)

    draw_panel(screen, (14, 246, 220, 152))
    draw_text(screen, font, "Camera", 26, 264)
    draw_text(screen, font, f"Yaw: {state.yaw:.2f}", 26, 296, muted)
    draw_text(screen, font, f"Pitch: {state.pitch:.2f}", 26, 322, muted)
    draw_text(screen, font, f"Distance: {state.distance:.2f}", 26, 348, muted)
    draw_text(screen, font, "Mouse Drag: Orbit", 26, 374, muted)

    draw_panel(screen, (14, 412, 220, 330))
    draw_text(screen, font, "Controls", 26, 430)

    controls = [
        ("Mouse Drag", "Orbit"),
        ("Scroll", "Zoom"),
        ("A / D", "Orbit"),
        ("Q / E", "Above/Below"),
        ("W / S", "Near/Far"),
        ("G", "Grid"),
        ("H", "Hide UI"),
        ("R", "Reset"),
        ("ESC", "Quit"),
    ]

    y = 464
    for k, v in controls:
        draw_text(screen, font, k, 26, y, muted)
        draw_text(screen, font, v, 116, y)
        y += 28

    draw_panel(screen, (1328, 50, 194, 214))
    draw_text(screen, font, "Visualization", 1342, 68)
    draw_checkbox(screen, font, "Accretion Disk", state.show_disk, 1344, 100)
    draw_checkbox(screen, font, "Photon Ring", state.show_ring, 1344, 130)
    draw_checkbox(screen, font, "Spacetime Grid", state.show_grid, 1344, 160)
    draw_checkbox(screen, font, "Stars", state.show_stars, 1344, 190)

    draw_panel(screen, (1328, 310, 194, 292))
    draw_text(screen, font, "Parameters", 1342, 328)
    draw_slider(screen, font, "BH Mass", state.mass, 1342, 360, 4.0, 20.0)
    draw_slider(screen, font, "Spin (a)", state.spin, 1342, 410, 0.0, 1.0)
    draw_slider(screen, font, "Accretion", state.accretion, 1342, 460, 0.0, 1.5)
    draw_slider(screen, font, "Time Scale", state.time_scale, 1342, 510, 0.1, 3.0)

    draw_button(screen, font, "Reset to Default", (1342, 558, 166, 28))


def reset(state):
    state.mass = 9.87
    state.spin = 0.85
    state.accretion = 0.80
    state.time_scale = 1.0
    state.yaw = 0.0
    state.pitch = 0.08
    state.distance = 4.2


def set_slider(state, name, mx):
    table = {
        "mass": (1342, 4.0, 20.0),
        "spin": (1342, 0.0, 1.0),
        "accretion": (1342, 0.0, 1.5),
        "time_scale": (1342, 0.1, 3.0),
    }

    sx, mn, mxv = table[name]
    f = clamp((mx - sx) / 170, 0.0, 1.0)
    setattr(state, name, mn + f * (mxv - mn))


def handle_click(state, pos):
    x, y = pos

    checks = [
        ("show_disk", 1344, 100),
        ("show_ring", 1344, 130),
        ("show_grid", 1344, 160),
        ("show_stars", 1344, 190),
    ]

    for name, cx, cy in checks:
        if cx <= x <= cx + 170 and cy <= y <= cy + 20:
            setattr(state, name, not getattr(state, name))
            return

    sliders = [
        ("mass", 1342, 390),
        ("spin", 1342, 440),
        ("accretion", 1342, 490),
        ("time_scale", 1342, 540),
    ]

    for name, sx, sy in sliders:
        if sx <= x <= sx + 170 and sy - 10 <= y <= sy + 15:
            state.dragging_slider = name
            set_slider(state, name, x)
            return

    if 1342 <= x <= 1508 and 558 <= y <= 586:
        reset(state)
        return

    state.dragging = True


def main():
    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )

    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Real-Time Black Hole Simulator")

    ctx = moderngl.create_context()
    ctx.disable(moderngl.DEPTH_TEST)

    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

    vertices = np.array(
        [
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ],
        dtype="f4",
    )

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, "in_pos")

    ui_program = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader="""
        #version 330
        uniform sampler2D u_texture;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            fragColor = texture(u_texture, v_uv);
        }
        """,
    )

    ui_vao = ctx.simple_vertex_array(ui_program, vbo, "in_pos")
    ui_texture = ctx.texture((WIDTH, HEIGHT), 4)

    ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    state = State()

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 14, bold=True)

    t = 0.0
    running = True
    last_mouse = None

    while running:
        dt = clock.tick(FPS) / 60.0
        fps = clock.get_fps()
        t += dt * state.time_scale

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    state.show_grid = not state.show_grid
                elif event.key == pygame.K_h:
                    state.show_ui = not state.show_ui
                elif event.key == pygame.K_r:
                    reset(state)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    handle_click(state, event.pos)
                    last_mouse = event.pos

                elif event.button == 4:
                    state.distance = clamp(state.distance * 0.92, 2.2, 8.0)

                elif event.button == 5:
                    state.distance = clamp(state.distance / 0.92, 2.2, 8.0)

            elif event.type == pygame.MOUSEBUTTONUP:
                state.dragging = False
                state.dragging_slider = None
                last_mouse = None

            elif event.type == pygame.MOUSEMOTION:
                if state.dragging_slider:
                    set_slider(state, state.dragging_slider, event.pos[0])

                elif state.dragging and last_mouse:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]

                    state.yaw += dx * 0.006
                    state.pitch = clamp(state.pitch + dy * 0.004, -1.25, 1.25)

                    last_mouse = event.pos

        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            state.yaw -= 0.025
        if keys[pygame.K_d]:
            state.yaw += 0.025
        if keys[pygame.K_q]:
            state.pitch = clamp(state.pitch + 0.018, -1.25, 1.25)
        if keys[pygame.K_e]:
            state.pitch = clamp(state.pitch - 0.018, -1.25, 1.25)
        if keys[pygame.K_w]:
            state.distance = clamp(state.distance * 0.985, 2.2, 8.0)
        if keys[pygame.K_s]:
            state.distance = clamp(state.distance / 0.985, 2.2, 8.0)

        program["u_resolution"].value = (WIDTH, HEIGHT)
        program["u_time"].value = t
        program["u_mass"].value = state.mass
        program["u_spin"].value = state.spin
        program["u_accretion"].value = state.accretion
        program["u_yaw"].value = state.yaw
        program["u_pitch"].value = state.pitch
        program["u_distance"].value = state.distance
        program["u_show_disk"].value = 1.0 if state.show_disk else 0.0
        program["u_show_ring"].value = 1.0 if state.show_ring else 0.0
        program["u_show_grid"].value = 1.0 if state.show_grid else 0.0
        program["u_show_stars"].value = 1.0 if state.show_stars else 0.0

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        if state.show_ui:
            ui_surface.fill((0, 0, 0, 0))
            draw_ui(ui_surface, font, state, fps)

            ui_data = pygame.image.tostring(ui_surface, "RGBA", True)
            ui_texture.write(ui_data)
            ui_texture.use(0)
            ui_program["u_texture"].value = 0

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ui_vao.render()
            ctx.disable(moderngl.BLEND)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()