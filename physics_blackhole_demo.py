# physics_blackhole_demo.py
# Real-time black-hole visualization using Pygame, ModernGL, and a procedural
# star-only sky. The script renders a black-hole shadow, photon-ring structure,
# accretion flow, spacetime-grid guide, UI controls, screenshots, and benchmarks.
#
# Dependencies:
#   pip install pygame moderngl numpy
#
# Run:
#   python physics_blackhole_demo_repo_stars_only_github.py
#
# This file is self-contained. It does not need a Milky Way image, EXR panorama,
# or other large texture asset; the sky is generated in the shader from compact
# procedural star layers.
#
# Scientific scope:
# This is a physically inspired, real-time educational visualization. It is not a
# full Kerr geodesic solver, numerical-relativity code, EHT reconstruction, or
# GRMHD radiative-transfer simulation. The shader keeps the render interactive by
# using fast approximations for gravitational bending, disk emission, and lensing.
#
# Physical ideas used by the renderer:
#   - Kerr-inspired horizon and ISCO radii are computed on the CPU.
#   - Scene-space disk radii are mapped to physical radii in units of GM/c^2.
#   - Orbital motion follows a Kerr-like Keplerian angular frequency.
#   - Redshift and Doppler effects are summarized by a frequency-shift factor g.
#   - Specific intensity is modulated with an approximate I_nu proportional to g^3.
#   - The accretion texture is guided by thin-disk temperature, optical-depth,
#     magnetic-field, and synchrotron-emission cues.
#   - Photon-ring highlights are tied to a critical-impact-parameter proxy.
#
# Controls and modes:
#   1 / 2 / 3 / 4  choose cinematic, balanced, strong-physics, or diagnostic mode
#   5 / 6          public-demo camera presets
#   7              physics-explanation preset with diagnostic overlay
#   8              cinematic export preset with UI hidden
#   P              save the visible frame
#   O              save a clean frame without UI panels
#   B              benchmark the current mode and export timing data
#   J / K          decrease / increase procedural star detail

import sys
import math
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import moderngl
import numpy as np
import pygame


WIDTH, HEIGHT = 1536, 864
FPS = 60

EXPORT_DIR = Path(__file__).with_name("blackhole_exports")
BENCHMARK_FRAMES = 180


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
uniform float u_isco;
uniform float u_horizon_kerr;
uniform float u_accretion;
uniform float u_yaw;
uniform float u_pitch;
uniform float u_distance;

uniform float u_show_disk;
uniform float u_show_ring;
uniform float u_show_grid;
uniform float u_show_stars;
uniform float u_show_particles;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_bloom;
uniform float u_science_overlay;
uniform float u_science_strength;
uniform float u_star_detail;

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
    // ACES-style filmic tone mapping compresses high dynamic range into the
    // displayable 0..1 range while keeping bright disk regions smooth.
    c = max(c, vec3(0.0));
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
    return pow(clamp(c, 0.0, 1.0), vec3(0.92));
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
    return normalize(cam * vec3(p, 1.88));
}

vec3 stellarColor(float t) {
    // Approximate stellar color-temperature palette. Hotter stars are blue-white,
    // Sun-like stars are pale white, and cooler stars are warmer orange.
    vec3 blueWhite = vec3(0.58, 0.70, 1.00);
    vec3 solar     = vec3(1.00, 0.96, 0.82);
    vec3 warm      = vec3(1.00, 0.70, 0.38);
    vec3 c = mix(blueWhite, solar, smoothstep(0.10, 0.66, t));
    return mix(c, warm, smoothstep(0.78, 1.0, t));
}

vec3 separatedStarLayer(vec2 uv, float scale, float threshold, float corePx,
                        float haloPx, float maxBrightness, float density,
                        float detail, float seed) {
    // One possible star is placed in each sky cell. A high threshold keeps most
    // cells empty, and a small random offset prevents the sky from looking like a grid.
    vec2 p = uv * vec2(scale, scale * 0.54);
    vec2 id = floor(p);
    vec2 f = fract(p);

    float r = hash21(id + seed);
    float localThreshold = threshold - detail * 0.00125 - (density - 1.0) * 0.00035;
    float starMask = step(localThreshold, r);

    vec2 offset = vec2(hash21(id + seed + 13.7), hash21(id + seed + 29.1)) - 0.5;
    vec2 q = f - 0.5 - offset * 0.24;
    float d2 = dot(q, q);

    // Pixel-scaled Gaussian kernels keep each star compact. The fwidth() term
    // estimates screen-space pixel size so star cores stay readable but not bloated.
    float px = max(max(fwidth(p.x), fwidth(p.y)), 0.00085);
    float coreSigma = max(px * corePx, 0.0016);
    float pinSigma = max(coreSigma * 0.52, 0.0011);
    float haloSigma = max(px * haloPx, coreSigma * 1.65);

    float pin = exp(-0.5 * d2 / (pinSigma * pinSigma));
    float core = exp(-0.5 * d2 / (coreSigma * coreSigma));
    float halo = exp(-0.5 * d2 / (haloSigma * haloSigma)) * 0.010;

    float temp = hash21(id + seed + 5.4);
    vec3 col = stellarColor(temp);

    // Stellar magnitudes are distributed so faint stars are common and bright
    // stars are rare, matching the way real star fields are visually dominated by
    // many small points and a few brighter objects.
    float luminosityRand = hash21(id + seed + 61.0);
    float magnitude = pow(luminosityRand, 4.2);
    float brightness = mix(maxBrightness * 0.22, maxBrightness, magnitude);
    brightness *= mix(0.88, 1.18, detail) * (0.96 + 0.08 * density);

    return starMask * col * brightness * (pin * 0.55 + core * 1.05 + halo);
}

vec3 starfield(vec3 rd) {
    if (u_show_stars < 0.5) return vec3(0.0);

    vec3 d = normalize(rd);

    // The sky direction is gently rotated before projection. This gives the star
    // pattern a stable orientation without relying on an external panorama.
    float yaw = -0.34 + 0.008 * sin(u_time * 0.014);
    float roll = 0.58;
    mat3 ry = mat3(
        cos(yaw), 0.0, sin(yaw),
        0.0,      1.0, 0.0,
       -sin(yaw), 0.0, cos(yaw)
    );
    mat3 rz = mat3(
        cos(roll), -sin(roll), 0.0,
        sin(roll),  cos(roll), 0.0,
        0.0,        0.0,       1.0
    );
    vec3 skyDir = normalize(rz * ry * d);

    vec2 uv = vec2(
        atan(skyDir.z, skyDir.x) / (2.0 * PI) + 0.5,
        asin(skyDir.y) / PI + 0.5
    );

    float detail = clamp(u_star_detail, 0.0, 1.0);

    // Deep space is not perfectly flat black. A very faint blue-black floor helps
    // the small procedural stars sit naturally in the background.
    float skyMottle = fbm(uv * vec2(3.2, 1.7) + vec2(0.21, 0.37));
    float densityNoise = fbm(uv * vec2(4.2, 2.0) + vec2(-1.7, 0.4));
    float density = mix(0.94, 1.06, smoothstep(0.18, 0.92, densityNoise));

    vec3 col = vec3(0.0011, 0.0015, 0.0032);
    col += vec3(0.0010, 0.0013, 0.0024) * skyMottle * 0.10;

    // Multiple star layers simulate magnitude classes: coarse layers create the
    // rare brighter stars, while finer layers add many distant pinpoints.
    col += separatedStarLayer(uv,   80.0, 0.944, 1.35, 2.10, 0.82, density, detail,  11.0);
    col += separatedStarLayer(uv,  145.0, 0.958, 1.12, 1.85, 0.64, density, detail,  37.0);
    col += separatedStarLayer(uv,  275.0, 0.974, 0.92, 1.55, 0.45, density, detail,  73.0);
    col += separatedStarLayer(uv,  540.0, 0.988, 0.78, 1.32, 0.29, density, detail, 131.0);
    col += separatedStarLayer(uv,  960.0, 0.994, 0.68, 1.15, 0.20, density, detail, 211.0);
    col += separatedStarLayer(uv, 1500.0, 0.997, 0.58, 1.05, 0.13, density, detail, 307.0);

    // A final sparse layer adds occasional brighter foreground stars without
    // turning them into large glowing blobs.
    col += separatedStarLayer(uv,  115.0, 0.989, 1.45, 2.30, 1.05, density, detail, 401.0) * 0.75;

    return col;
}
float gridLine(float x, float w) {
    float g = abs(fract(x) - 0.5);
    return 1.0 - smoothstep(w * 0.35, w, g);
}

vec3 sampleGrid(vec3 p) {
    if (u_show_grid < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float r = length(p.xz) + 0.0001;
    if (r < horizon * 1.08) return vec3(0.0);

    // Visual embedding diagram for the spacetime grid. The sheet is pulled down
    // near the hole with a simple well term, z_well ~ -K/(r + offset), so the grid
    // communicates curvature without acting like physical plasma emission.
    float well = -0.82 * (horizon * 2.10) / (r + horizon * 1.20);
    float d = abs(p.y - well);

    float sheetCore = exp(-(d * d) / (2.0 * 0.026 * 0.026));
    float sheetSoft = exp(-d * 15.0) * 0.11;
    float sheet = sheetCore + sheetSoft;

    float pull = 0.38 * horizon * horizon / (r * r + horizon * horizon);
    vec2 q = p.xz * (1.0 - pull);

    float spacing = 0.32;
    float gx = gridLine(q.x / spacing, 0.055);
    float gz = gridLine(q.y / spacing, 0.055);
    float majorX = gridLine(q.x / (spacing * 2.0), 0.035);
    float majorZ = gridLine(q.y / (spacing * 2.0), 0.035);

    float minor = max(gx, gz) * 0.58;
    float major = max(majorX, majorZ) * 0.80;
    float line = max(minor, major);

    float fadeOuter = 1.0 - smoothstep(1.2, 7.2, r);
    float fadeInner = smoothstep(horizon * 1.10, horizon * 2.10, r);
    float wellGlow = exp(-pow(r - horizon * 1.75, 2.0) / 0.50) * 0.055;

    vec3 minorCol = vec3(0.040, 0.078, 0.105);
    vec3 majorCol = vec3(0.065, 0.135, 0.175);
    vec3 col = mix(minorCol, majorCol, smoothstep(0.45, 0.88, major));

    return (col * line * sheet * 1.22 + vec3(0.030, 0.080, 0.105) * wellGlow * sheet)
           * fadeOuter * fadeInner;
}

float visualMassScale() {
    // The mass slider affects the physical readout and lensing strength, but the
    // visual scale is softly clamped. Apparent black-hole size depends on both mass
    // and camera distance, so this keeps interactive framing stable.
    return clamp(u_mass / 9.87, 0.80, 1.15);
}

float sceneDiskInnerRadius(float visualHorizon) {
    // Kerr prograde ISCO radius is supplied by the CPU in units of GM/c^2. Higher
    // spin moves the stable inner disk edge inward, and this function maps that
    // physical trend into the scene scale.
    float t = clamp((u_isco - 1.237) / (6.0 - 1.237), 0.0, 1.0);
    return visualHorizon * mix(1.14, 1.24, t);
}

float physicalDiskRadius(float rScene, float innerScene) {
    // Convert scene radius into an approximate physical radius in GM/c^2. This
    // hidden coordinate drives orbital speed, disk temperature, and redshift.
    return max(u_horizon_kerr + 0.05, u_isco * rScene / max(innerScene, 0.001));
}

float keplerianOmega(float rPhys) {
    // Kerr-like prograde circular-orbit frequency in geometrized units:
    // omega = 1 / (r^(3/2) + a), where a is the dimensionless spin.
    return 1.0 / (pow(max(rPhys, 1.001), 1.5) + clamp(u_spin, 0.0, 0.998));
}

float thinDiskTemperature(float rScene, float innerScene) {
    // Thin-disk temperature profile inspired by Shakura-Sunyaev and Novikov-Thorne
    // disks. The r^(-3/4) trend makes inner material hotter; the no-torque factor
    // softens emission at the ISCO. The result is normalized for color, not Kelvin.
    float x = max(rScene / max(innerScene, 0.001), 1.001);
    float noTorque = pow(max(1.0 - inversesqrt(x), 0.0), 0.25);
    return clamp(pow(x, -0.75) * noTorque * 2.55, 0.0, 1.20);
}

float redshiftFactor(vec3 orbitalDir, vec3 rd, float rPhys) {
    // Approximate frequency-shift factor g = nu_observed / nu_emitted. It combines
    // special-relativistic Doppler shift from orbital motion with a simple
    // gravitational-redshift term near the black hole.
    float a = clamp(u_spin, 0.0, 0.998);
    float beta = clamp(sqrt(1.0 / max(rPhys, 2.20)) * (0.56 + 0.10 * a), 0.025, 0.62);
    float gamma = inversesqrt(max(0.001, 1.0 - beta * beta));
    float mu = dot(orbitalDir, -rd);
    float doppler = 1.0 / max(0.12, gamma * (1.0 - beta * mu));
    float grav = sqrt(clamp(1.0 - 1.62 / max(rPhys, 1.82), 0.22, 1.0));
    return clamp(doppler * grav, 0.34, 2.20);
}

float relativisticBeaming(vec3 orbitalDir, vec3 rd, float rPhys) {
    // Relativistic beaming uses the invariant transfer relation I_nu / nu^3.
    // This gives the familiar I_nu proportional to g^3 brightness change, with
    // clamps so the interactive image remains readable.
    float g = redshiftFactor(orbitalDir, rd, rPhys);

    float beamStrength = mix(0.58, 0.82, u_science_strength);
    float minBeam = mix(0.42, 0.34, u_science_strength);
    float maxBeam = mix(3.55, 4.40, u_science_strength);

    return clamp(mix(1.0, pow(g, 3.0), beamStrength), minBeam, maxBeam);
}

float criticalImpactParam() {
    // Scene-space proxy for the photon critical curve. Rays near this impact
    // parameter skim the photon region and contribute to the thin bright ring.
    float horizon = 0.62;
    return horizon * (1.18 - 0.022 * clamp(u_spin, 0.0, 0.998));
}

float geodesicWindingFromImpact(float b) {
    // Near the critical impact parameter, null rays can wind around the hole.
    // A logarithmic term approximates that rapid growth in path length near the
    // photon region and fades higher-order features.
    float bc = criticalImpactParam();
    float eps = abs(b - bc) / max(bc, 0.0001);
    return clamp(-log(max(eps, 0.0016)) * 0.46, 0.0, 3.8);
}

float photonCriticalWeight(float b, float width) {
    float bc = criticalImpactParam();
    float critical = exp(-pow((b - bc) / max(width, 0.0001), 2.0));
    float winding = geodesicWindingFromImpact(b);
    return critical * exp(-0.62 * winding);
}

float synchrotronTransferWeight(float ne, float thetaE, float bMag, float pitchAngle) {
    // Lightweight optically thin synchrotron emissivity model:
    // j_nu ~ n_e B^(3/2) exp[-sqrt(nu / nu_c)]. It modulates disk texture using
    // density, magnetic field strength, electron temperature, and pitch angle.
    float nu = 1.0;
    float nuC = 0.050 + 0.52 * bMag * thetaE * thetaE * pitchAngle;
    float jnu = ne * pow(max(bMag * pitchAngle, 0.001), 1.5) * exp(-sqrt(nu / max(nuC, 0.0001)));
    return clamp(0.75 + 0.45 * jnu, 0.70, 1.35);
}

float opticalDepthCue(float ne, float thetaE, float bMag) {
    // Optical-depth cue for absorption. Dense, cooler lanes reduce emission a
    // little, giving the disk dark structure instead of uniform glow.
    float alpha = ne * sqrt(max(bMag, 0.001)) / (thetaE * thetaE + 0.30);
    return clamp(alpha, 0.0, 1.0);
}

vec3 advanceNullGeodesic(vec3 pos, vec3 dir, float ds) {
    // Fast weak-field null-ray step. Gravity changes only the transverse part of
    // the photon direction, which preserves the idea that light is bent sideways
    // toward the mass rather than accelerated along its own direction.
    float horizon = 0.62;
    float r = length(pos) + 0.0001;
    float b = length(cross(pos, dir)) + 0.0001;
    float massScale = visualMassScale();

    vec3 towardBH = -pos / r;
    vec3 transverse = towardBH - dir * dot(towardBH, dir);
    float tLen = length(transverse);
    if (tLen > 0.0001) {
        transverse /= tLen;
    }

    float photonShell = exp(-pow(r - horizon * 1.50, 2.0) / 0.078);
    float nearShell = exp(-pow(r - horizon * 1.11, 2.0) / 0.030);
    float critical = photonCriticalWeight(b, horizon * 0.22);

    float bend = 0.040 * massScale / (r * r + 0.18);
    bend += 0.044 * photonShell + 0.018 * nearShell + 0.010 * critical / (r + 0.44);
    bend *= mix(1.0, 1.18, u_science_strength);

    vec3 spinAxis = vec3(0.0, 1.0, 0.0);
    vec3 drag = cross(spinAxis, pos);
    drag -= dir * dot(drag, dir);
    float dLen = length(drag);
    vec3 frameDrag = dLen > 0.0001 ? drag / dLen : vec3(0.0);
    float dragAmount = 0.0052 * clamp(u_spin, 0.0, 0.998) * massScale / (r * r + 0.42);
    dragAmount *= mix(1.0, 1.28, u_science_strength);

    return normalize(dir + transverse * bend * ds + frameDrag * dragAmount * ds);
}

vec3 sampleDisk(vec3 p, vec3 rd) {
    if (u_show_disk < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float inner = sceneDiskInnerRadius(horizon);
    float outer = 3.35;

    if (length(p) < horizon * 1.03) return vec3(0.0);

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

    // Geometrically thin emitting disk. The Gaussian vertical profile confines
    // most light to the disk plane, while a faint lifted ribbon represents the
    // secondary lensed image of disk material behind the black hole.
    float qDisk = max(r - inner, 0.0);
    float thickness = 0.064 + 0.034 * exp(-1.22 * qDisk) + 0.010 * smoothstep(1.25, 2.8, r);
    float vertical = exp(-(y * y) / (2.0 * thickness * thickness));
    float lensLift = 0.115 + 0.145 * exp(-pow(r - horizon * 1.52, 2.0) / 0.22);
    float lensedSkin = exp(-pow(y - lensLift, 2.0) / (2.0 * (thickness * 0.72) * (thickness * 0.72)))
                     * exp(-pow(r - horizon * 1.58, 2.0) / 0.30) * 0.165;
    float corona = exp(-abs(y) / (thickness * 1.70)) * exp(-0.95 * qDisk) * 0.050;
    vertical = max(vertical, lensedSkin + corona);

    // Dark central cavity. Material inside the inner stable region contributes
    // little visible disk emission, so the black-hole shadow stays clean.
    float cavity = smoothstep(inner, inner + 0.23, r);
    float radial = exp(-1.05 * max(r - inner, 0.0)) * cavity;

    float rPhys = physicalDiskRadius(r, inner);
    float omega = keplerianOmega(rPhys);
    float tempPhys = thinDiskTemperature(r, inner);

    float swirl = a + 0.95 / (r + 0.12) + u_time * (1.85 * omega + 0.10 + 0.18 * u_spin);
    vec2 flowUV = vec2(r * 6.8 - u_time * (0.16 + 0.38 * omega), swirl * 4.25);
    float turbulent = fbm(flowUV);
    float fineDust = fbm(flowUV * 3.7 + vec2(4.1, -1.7));
    float microDust = fbm(flowUV * 9.8 + vec2(-2.3, 5.2));
    float shear = sin(92.0 * r - 16.0 * a - u_time * (3.7 + 1.5 * u_spin));
    float filamentNoise = smoothstep(0.46, 0.92, microDust) * (0.42 + 0.58 * abs(shear));

    // Narrow luminous rings emphasize hot inner accretion structure and the
    // photon-orbit region where lensing strongly concentrates light.
    float ring1 = exp(-pow(r - 0.88, 2.0) / 0.0038);
    float ring2 = exp(-pow(r - 1.08, 2.0) / 0.0070);
    float ring3 = exp(-pow(r - 1.42, 2.0) / 0.0180);
    float ringTexture = 0.72 + 0.20 * sin(96.0 * r + 11.0 * a - u_time * 2.4) + 0.16 * turbulent + 0.12 * filamentNoise;
    float rings = (ring1 * 1.30 + ring2 * 0.62 + ring3 * 0.28) * ringTexture;

    float armA = sin(22.0 * r - 5.8 * a - u_time * 1.85 + turbulent * 3.4);
    float armB = sin(44.0 * r + 3.9 * a - u_time * 1.20 + fineDust * 2.2);
    float armC = sin(82.0 * r - 9.0 * a - u_time * 2.60 + microDust * 1.4);
    float streaks = smoothstep(0.20, 1.0, abs(armC)) * filamentNoise;
    float spiral = 0.50 + 0.20 * turbulent + 0.12 * armA + 0.075 * armB + 0.050 * streaks + rings;

    // Absorbing lanes inside the accretion flow add cooler, darker texture to the
    // turbulent disk without filling the central shadow.
    float lane = smoothstep(0.58, 0.94, fineDust) * (0.35 + 0.65 * smoothstep(1.05, 2.2, r));
    spiral *= 1.0 - lane * 0.22;
    spiral += filamentNoise * (1.0 - smoothstep(0.82, 2.9, r)) * 0.18;

    // Small hot clumps represent short-lived turbulent density enhancements inside
    // the disk and inner rings.
    vec2 cell = vec2(a * 34.0 + r * 8.0 - u_time * 0.70, r * 24.0);
    vec2 cid = floor(cell);
    vec2 cf = fract(cell);
    float cr = hash21(cid + 44.2);
    float clumpMask = smoothstep(0.946, 1.0, cr);
    vec2 coff = vec2(hash21(cid + 2.0), hash21(cid + 7.0));
    float cdist = length(cf - coff);
    float ringPreference = ring1 * 0.90 + ring2 * 0.70 + (1.0 - smoothstep(0.95, 2.6, r)) * 0.35;
    float clumps = clumpMask * exp(-cdist * 24.0) * ringPreference;

    vec3 orbitalDir = normalize(-sin(a) * diskX + cos(a) * diskZ);
    vec3 radialDir = normalize(cos(a) * diskX + sin(a) * diskZ);
    float g = redshiftFactor(orbitalDir, rd, rPhys);
    float doppler = relativisticBeaming(orbitalDir, rd, rPhys);

    // Magnetic and synchrotron-inspired modulation. The field is mostly toroidal
    // with a weaker poloidal part, and the pitch angle controls how efficiently
    // electrons radiate toward the camera.
    float toroidalStrength = 0.88 + 0.10 * clamp(u_spin, 0.0, 0.998);
    float poloidalStrength = 0.13 + 0.09 * smoothstep(0.20, 1.80, qDisk) + 0.025 * sin(2.5 * a + u_time * 0.18);
    vec3 bField = normalize(orbitalDir * toroidalStrength + diskNormal * poloidalStrength + radialDir * (0.045 * (turbulent - 0.5)));
    float pitchAngle = clamp(length(cross(bField, -rd)), 0.065, 1.0);
    float ne = vertical * radial * (0.54 + 0.46 * turbulent + 0.28 * ring1 + 0.16 * ring2);
    float thetaE = clamp(0.32 + 1.50 * pow(tempPhys, 1.70) + 0.18 * ring1 + 0.08 * filamentNoise, 0.10, 2.60);
    float bMag = pow(max(rPhys, 1.15), -1.04) * (1.0 + 0.36 * ring1 + 0.14 * ring2 + 0.08 * filamentNoise);
    float synchMod = synchrotronTransferWeight(ne, thetaE, bMag, pitchAngle);
    float tau = opticalDepthCue(ne, thetaE, bMag);
    float redshiftMod = clamp(pow(g, 3.0), 0.55, 1.85);

    float heat = mix(exp(-0.68 * max(r - inner, 0.0)), tempPhys, 0.18);
    float photonBoost = 1.0 + 0.55 * ring1 + 0.28 * ring2;
    float redshiftTransferStrength = mix(0.28, 0.52, u_science_strength);
    float absorptionStrength = mix(0.075, 0.140, u_science_strength);
    float physicalMod = synchMod * mix(1.0, redshiftMod, redshiftTransferStrength) * (1.0 - tau * lane * absorptionStrength);

    float brightness = vertical * radial * spiral * doppler * heat * photonBoost * physicalMod * u_accretion * 1.22;
    brightness += vertical * clumps * doppler * heat * physicalMod * 1.48 * u_accretion;
    brightness += vertical * filamentNoise * radial * doppler * heat * 0.25 * physicalMod * u_accretion;
    brightness = pow(max(brightness, 0.0), 0.56);

    vec3 base = hotColor(clamp(brightness + tempPhys * 0.10 + max(g - 1.0, 0.0) * 0.035, 0.0, 1.0)) * brightness;

    vec3 whiteRim = vec3(1.0, 0.94, 0.76) * vertical * ring1 * doppler * physicalMod * 0.74 * u_accretion;
    vec3 ringSpark = vec3(1.0, 0.84, 0.55) * clumps * vertical * doppler * physicalMod * 0.64;
    vec3 coldLane = vec3(0.22, 0.095, 0.040) * vertical * radial * lane * (0.09 + tau * 0.035);
    vec3 blueShock = vec3(0.55, 0.68, 1.0) * clumps * max(g - 1.16, 0.0) * pitchAngle * 0.18;

    return base + whiteRim + ringSpark + blueShock + coldLane;
}

vec3 samplePhotonHalo(vec3 p, vec3 rd) {
    if (u_show_ring < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float inner = 0.70;
    float outer = 2.05;

    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(p, diskX);
    float y = dot(p, diskNormal);
    float z = dot(p, diskZ);
    float r = length(vec2(x, z)) + 0.0001;
    float a = atan(z, x);
    float sphericalR = length(p);

    if (sphericalR < horizon * 1.02 || r < inner || r > outer) return vec3(0.0);

    // Curved lensed photon ribbon. Disk light passing close to the photon region
    // forms connected arcs around the shadow instead of a separate jet-like beam.
    float equatorialRibbon = exp(-(y * y) / (2.0 * 0.076 * 0.076));
    float arcLift = 0.070 + 0.245 * exp(-pow(r - critical * 1.305, 2.0) / 0.25);
    float arcWidth = 0.045 + 0.018 * smoothstep(horizon * 1.12, horizon * 2.00, r);
    float upperRibbon = exp(-pow(y - arcLift, 2.0) / (2.0 * arcWidth * arcWidth))
                      * exp(-pow(r - critical * 1.34, 2.0) / 0.32) * 0.72;
    float lowerRibbon = exp(-pow(y + arcLift * 0.52, 2.0) / (2.0 * (arcWidth * 0.88) * (arcWidth * 0.88)))
                      * exp(-pow(r - critical * 1.23, 2.0) / 0.28) * 0.26;
    float sideBridge = exp(-(y * y) / (2.0 * 0.115 * 0.115))
                     * exp(-pow(r - critical * 1.20, 2.0) / 0.115) * 0.38;
    float azimuthContinuity = 0.88 + 0.12 * cos(2.0 * a);
    float vertical = (equatorialRibbon * 0.88 + sideBridge + upperRibbon + lowerRibbon) * azimuthContinuity;

    float winding = geodesicWindingFromImpact(r);
    float photon = exp(-pow(r - critical, 2.0) / 0.0036) * (0.94 + 0.06 * winding);
    float innerFire = exp(-pow(r - critical * 1.17, 2.0) / 0.0105);
    float midBand = exp(-pow(r - critical * 1.39, 2.0) / 0.0260);
    float outerBand = exp(-pow(r - critical * 1.76, 2.0) / 0.0740) * exp(-0.35 * winding);

    float rPhys = max(u_horizon_kerr + 0.05, 2.0 + (r / horizon - 1.0) * 2.4);
    float omega = keplerianOmega(rPhys);
    float swirl = a + 1.15 / (r + 0.11) + u_time * (3.2 * omega + 0.24 + 0.22 * u_spin);
    vec2 uv = vec2(r * 12.0 - u_time * 0.42, swirl * 7.8);
    float n1 = fbm(uv);
    float n2 = fbm(uv * 2.7 + vec2(3.7, -2.2));
    float n3 = fbm(uv * 7.8 + vec2(-1.3, 4.1));

    float strandA = 0.5 + 0.5 * sin(122.0 * r + 19.0 * a - u_time * 3.8 + n1 * 4.0);
    float strandB = 0.5 + 0.5 * sin(207.0 * r - 31.0 * a - u_time * 5.4 + n2 * 2.8);
    float filaments = smoothstep(0.28, 0.96, strandA) * (0.55 + 0.45 * n1)
                    + smoothstep(0.62, 0.99, strandB) * (0.35 + 0.65 * n2);
    filaments *= 0.58 + 0.42 * smoothstep(0.42, 0.92, n3);

    vec3 orbitalDir = normalize(-sin(a) * diskX + cos(a) * diskZ);
    float g = redshiftFactor(orbitalDir, rd, rPhys);
    float doppler = relativisticBeaming(orbitalDir, rd, rPhys);
    float criticalWeight = photonCriticalWeight(r, critical * 0.10);

    float shell = (photon * 1.30 + innerFire * 0.78 + midBand * 0.46 + outerBand * 0.14);
    shell *= 0.94 + 0.14 * criticalWeight;
    float haloRedshiftStrength = mix(0.22, 0.42, u_science_strength);
    float brightness = vertical * shell * (0.54 + 0.58 * filaments) * doppler * mix(1.0, clamp(pow(g, 3.0), 0.60, 1.75), haloRedshiftStrength) * u_accretion;

    // Small plasma sparks add granular structure to the photon-halo region and
    // help the ring feel like turbulent emitting material.
    vec2 cell = vec2(a * 78.0 + r * 13.0 - u_time * (1.05 + 0.45 * u_spin), r * 44.0 + y * 8.0);
    vec2 id = floor(cell);
    vec2 f = fract(cell);
    vec2 off = vec2(hash21(id + 5.1), hash21(id + 17.3));
    float rnd = hash21(id + 41.7);
    float bead = smoothstep(0.972, 1.0, rnd) * exp(-length(f - off) * 34.0);
    bead *= (photon * 1.0 + innerFire * 0.85 + midBand * 0.62) * vertical * u_show_particles;

    vec3 col = vec3(1.0, 0.56, 0.18) * brightness * 0.92;
    col += vec3(1.0, 0.90, 0.62) * photon * vertical * doppler * (0.31 + 0.15 * filaments) * u_accretion;
    col += vec3(1.0, 0.78, 0.38) * bead * doppler * 1.38 * u_accretion;
    col += vec3(1.0, 0.31, 0.08) * outerBand * vertical * filaments * 0.095 * doppler * u_accretion;

    // The central attenuation keeps the shadow dark, making the bright ring read
    // as surrounding the horizon rather than filling it.
    col *= smoothstep(horizon * 1.04, horizon * 1.34, sphericalR);
    return col;
}

vec3 sampleLensRim(vec3 ro, vec3 rd, vec2 p, float closestImpact) {
    if (u_show_ring < 0.5) return vec3(0.0);

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float winding = geodesicWindingFromImpact(closestImpact);
    vec3 closest = ro + rd * dot(-ro, rd);

    float tilt = 0.18;
    vec3 diskNormal = normalize(vec3(0.0, cos(tilt), sin(tilt)));
    vec3 diskX = vec3(1.0, 0.0, 0.0);
    vec3 diskZ = normalize(cross(diskX, diskNormal));

    float x = dot(closest, diskX);
    float y = dot(closest, diskNormal);
    float z = dot(closest, diskZ);
    float a = atan(z, x);
    float diskSide = dot(normalize(cross(rd, diskX) + vec3(0.0001)), diskNormal);
    float theta = atan(y * 1.15, x);
    float ringCoord = theta / (2.0 * PI) + 0.5;

    float noiseArc = fbm(vec2(ringCoord * 28.0 + u_time * 0.11, closestImpact * 18.0));
    float broken = smoothstep(0.18, 0.98, 0.5 + 0.5 * sin(148.0 * closestImpact + 17.0 * a - u_time * 2.4 + noiseArc * 4.2));
    broken *= 0.55 + 0.45 * noiseArc;

    float rim = exp(-pow(closestImpact - critical * 0.915, 2.0) / 0.0028) * (0.94 + 0.06 * winding);
    float innerFire = exp(-pow(closestImpact - critical * 1.085, 2.0) / 0.0110);
    float outerArc = exp(-pow(closestImpact - critical * 1.373, 2.0) / 0.0500) * exp(-0.18 * winding);

    // Rim brightness is based on the ray's closest world-space pass around the
    // hole and its position in the disk frame.
    float diskFrameGate = smoothstep(0.02, 0.62, abs(x)) * (0.72 + 0.28 * smoothstep(-0.35, 0.35, y * diskSide));
    float asymStrength = mix(0.36, 0.48, u_science_strength);
    float asym = 1.0 + asymStrength * tanh(-x * 1.4 + u_spin * 0.45);

    vec3 col = vec3(0.0);
    col += vec3(1.0, 0.94, 0.76) * rim * (0.84 + 0.16 * broken) * 0.34 * u_accretion;
    col += vec3(1.0, 0.66, 0.25) * innerFire * (0.055 + 0.090 * broken) * diskFrameGate * asym * u_accretion;
    col += vec3(1.0, 0.40, 0.11) * outerArc * broken * diskFrameGate * asym * 0.058 * u_accretion;

    // The upper lensed arc is tied to the same impact-parameter ring and is only
    // slightly lifted from the disk plane, so it remains connected to the shadow.
    float liftedCenter = 0.090 + 0.080 * exp(-pow(closestImpact - critical * 1.203, 2.0) / 0.038);
    float upperArcTrack = exp(-pow(y - liftedCenter, 2.0) / (2.0 * 0.070 * 0.070));
    float connectedArc = upperArcTrack * exp(-pow(closestImpact - critical * 1.220, 2.0) / 0.030)
                       * smoothstep(-0.02, 0.18, y) * (0.68 + 0.32 * broken) * diskFrameGate;
    col += vec3(1.0, 0.78, 0.38) * connectedArc * asym * 0.070 * u_accretion;

    if (u_show_particles > 0.5) {
        vec2 cell = vec2(ringCoord * 92.0 + u_time * (0.75 + u_spin), closestImpact * 36.0 + a * 1.7);
        vec2 id = floor(cell);
        vec2 f = fract(cell);
        vec2 off = vec2(hash21(id + 8.0), hash21(id + 19.0));
        float rnd = hash21(id + 61.0);
        float spark = smoothstep(0.984, 1.0, rnd) * exp(-length(f - off) * 38.0);
        col += vec3(1.0, 0.84, 0.50) * spark * (rim * 0.50 + innerFire * 0.80 + outerArc * 0.62) * diskFrameGate * 1.10 * u_accretion;
    }

    return col;
}


vec3 scienceCalibrationOverlay(vec3 col, vec2 p, float impactScreen, float impactPath) {
    if (u_science_overlay < 0.5) return col;

    float horizon = 0.62;
    float critical = criticalImpactParam();
    float inner = sceneDiskInnerRadius(horizon);

    // Thin diagnostic guide curves. They are annotation markers for ISCO,
    // horizon scale, photon critical curve, and closest ray approach; they are
    // not treated as glowing physical plasma.
    float iscoMark = exp(-pow(impactScreen - inner, 2.0) / 0.0012);
    float horizonMark = exp(-pow(impactScreen - horizon, 2.0) / 0.0010);
    float photonMark = exp(-pow(impactScreen - critical, 2.0) / 0.0011);
    float pathMark = exp(-pow(impactPath - critical, 2.0) / 0.0035);

    vec3 overlay = vec3(0.0);
    overlay += vec3(0.18, 0.62, 1.0) * iscoMark * 0.22;       // ISCO guide
    overlay += vec3(1.0, 0.24, 0.12) * horizonMark * 0.18;    // horizon scale
    overlay += vec3(0.95, 0.84, 0.34) * photonMark * 0.30;    // photon critical curve
    overlay += vec3(0.28, 1.0, 0.66) * pathMark * 0.07;       // bent-path closest approach

    float stripe = smoothstep(0.48, 0.54, fract((p.y + 1.0) * 18.0));
    overlay *= 0.82 + 0.18 * stripe;
    return mix(col, clamp(col + overlay, 0.0, 1.0), 0.52);
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
    float pathMinImpact = 1e9;

    bool absorbed = false;

    float stepSize = 0.038;

    for (int i = 0; i < 232; i++) {
        float r = length(rayPos);
        float pathImpact = length(cross(rayPos, rayDir));
        pathMinImpact = min(pathMinImpact, pathImpact);

        if (r < horizon) {
            absorbed = true;
            break;
        }

        vec3 diskCol = sampleDisk(rayPos, rayDir);
        vec3 ringCol = samplePhotonHalo(rayPos, rayDir);
        vec3 gridCol = sampleGrid(rayPos);

        float localPower = length(diskCol) + length(ringCol) * 0.72;

        emission += diskCol * 0.071 * (1.0 - opacity);
        emission += ringCol * 0.086 * (1.0 - opacity * 0.62);
        opacity += localPower * 0.0080 * (1.0 - opacity);

        // The grid is an explanatory guide, so it remains visible through moderate
        // opacity while staying weaker than the bright accretion emission.
        emission += gridCol * 0.066 * (1.0 - opacity * 0.30);

        rayDir = advanceNullGeodesic(rayPos, rayDir, stepSize);
        rayPos += rayDir * stepSize;

        if (length(rayPos) > 9.0 || opacity > 0.96) break;
    }

    vec3 col = emission;

    float closestImpact = length(cross(ro, rd));
    float physicalClosestImpact = min(closestImpact, pathMinImpact);
    float criticalImpact = criticalImpactParam();

    // Near the shadow, gravitational absorption suppresses diffuse background
    // light while still allowing compact stars to appear through the lensing field.
    float shadowHalo = exp(-pow(closestImpact - horizon * 1.28, 2.0) / 0.19);
    float deepShadow = exp(-pow(closestImpact - horizon * 0.82, 2.0) / 0.035);

    if (!absorbed) {
        vec3 bg = starfield(rayDir) * 1.95;
        // Photon-sphere attenuation dims background stars near the critical region,
        // helping the lensed star field curve around the black hole.
        bg *= 1.0 - shadowHalo * 0.20;
        bg *= 1.0 - deepShadow * 0.64;
        col += bg * (1.0 - opacity);
    }

    // Thin lensing rim tied to the ray's closest pass around the hole. The richer
    // 3D emission is handled during ray marching, while this adds a crisp edge cue.
    col += sampleLensRim(ro, rd, p, closestImpact);

    // Final local shadow shaping. The attenuation is kept near the hole so the
    // rest of the star field and disk remain clear.
    float bowl = exp(-pow(closestImpact - horizon * 1.20, 2.0) / 0.050);
    float ringProtect = exp(-pow(closestImpact - criticalImpact, 2.0) / 0.0080);
    col *= 1.0 - bowl * 0.34 * (1.0 - ringProtect * 0.72);

    if (absorbed) {
        // Once a ray falls inside the horizon, no background light is added. Any
        // already accumulated foreground disk or ring emission is kept.
        col *= 0.88;
    }

    float vignette = smoothstep(1.68, 0.18, length(p));
    col *= mix(0.48, 1.0, vignette);

    // Single-pass bloom approximation. Bright values above the white-hot threshold
    // add a soft glow, giving the disk an HDR feel without a multipass post-process.
    float preLuma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    vec3 hotBloom = max(col - vec3(0.92), vec3(0.0));
    col += hotBloom * (0.115 + 0.145 * u_bloom);
    col += vec3(1.0, 0.58, 0.18) * smoothstep(0.8, 2.8, preLuma) * 0.026 * u_bloom;
    col *= u_exposure;
    col = tonemap(col);
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(luma), col, u_saturation);
    col = (col - 0.5) * u_contrast + 0.5;
    col = clamp(col, 0.0, 1.0);

    // Apply diagnostic guides after tone mapping so the overlay behaves like a
    // screen annotation rather than a source of physical light.
    col = scienceCalibrationOverlay(col, p, closestImpact, physicalClosestImpact);

    fragColor = vec4(col, 1.0);
}

"""




@dataclass
class State:
    mass: float = 9.87
    spin: float = 0.85
    accretion: float = 0.82
    time_scale: float = 1.0

    yaw: float = 0.0
    pitch: float = 0.16
    distance: float = 5.8

    show_disk: bool = True
    show_ring: bool = True
    show_grid: bool = True
    show_stars: bool = True
    show_particles: bool = True
    cinematic: bool = False
    science_overlay: bool = False
    science_strength: float = 0.0
    star_detail: float = 0.68
    show_ui: bool = True

    status_message: str = ""
    status_until_ms: int = 0

    exposure: float = 1.30
    bloom: float = 1.55
    contrast: float = 1.06
    saturation: float = 0.94

    dragging: bool = False
    dragging_slider: str | None = None

def clamp(x, a, b):
    return max(a, min(b, x))


def kerr_horizon_radius(spin: float) -> float:
    """Outer Kerr horizon radius r_+ in units of GM/c^2."""
    a = clamp(spin, 0.0, 0.998)
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def kerr_isco_radius(spin: float) -> float:
    """Prograde Kerr ISCO radius in units of GM/c^2."""
    a = clamp(spin, 0.0, 0.998)
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    term = max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    return 3.0 + z2 - math.sqrt(term)



def draw_text(screen, font, txt, x, y, color=(235, 235, 235)):
    screen.blit(font.render(txt, True, color), (x, y))


def science_mode_label(value: float) -> str:
    if value < 0.10:
        return "Cinematic"
    if value < 0.55:
        return "Balanced"
    return "Science"


def science_mode_slug(value: float) -> str:
    return science_mode_label(value).lower().replace(" ", "_")


def set_status(state, message: str, seconds: float = 3.5) -> None:
    state.status_message = message
    state.status_until_ms = pygame.time.get_ticks() + int(seconds * 1000)


def export_dir() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return EXPORT_DIR


def export_stem(state, prefix: str, include_ui: bool | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = science_mode_slug(state.science_strength)
    overlay = "overlay" if state.science_overlay else "clean"
    if include_ui is None:
        ui = "ui" if state.show_ui else "no_ui"
    else:
        ui = "ui" if include_ui else "no_ui"
    return f"{prefix}_{stamp}_{mode}_{overlay}_{ui}"


def state_report_lines(state, sim_time: float, fps: float) -> list[str]:
    return [
        "Real-Time Black Hole Simulator export",
        f"mode_label: {science_mode_label(state.science_strength)}",
        f"science_strength: {state.science_strength:.3f}",
        f"star_detail: {state.star_detail:.3f}",
        f"science_overlay: {state.science_overlay}",
        f"mass_M_sun: {state.mass:.4f}",
        f"spin_a: {state.spin:.4f}",
        f"accretion_rate: {state.accretion:.4f}",
        f"time_scale: {state.time_scale:.4f}",
        f"sim_time_s: {sim_time:.4f}",
        f"fps_readout: {fps:.3f}",
        f"camera_yaw_rad: {state.yaw:.6f}",
        f"camera_pitch_rad: {state.pitch:.6f}",
        f"camera_distance: {state.distance:.6f}",
        f"show_disk: {state.show_disk}",
        f"show_ring: {state.show_ring}",
        f"show_grid: {state.show_grid}",
        f"show_stars: {state.show_stars}",
        f"show_particles: {state.show_particles}",
        f"show_ui: {state.show_ui}",
        f"exposure: {state.exposure:.4f}",
        f"bloom: {state.bloom:.4f}",
        f"contrast: {state.contrast:.4f}",
        f"saturation: {state.saturation:.4f}",
    ]


def save_framebuffer(ctx, state, sim_time: float, fps: float, include_ui: bool) -> Path:
    path = export_dir() / f"{export_stem(state, 'blackhole_screenshot', include_ui)}.png"
    ctx.finish()
    data = ctx.screen.read(components=3, alignment=1)
    surface = pygame.image.frombuffer(data, (WIDTH, HEIGHT), "RGB").copy()
    surface = pygame.transform.flip(surface, False, True)
    pygame.image.save(surface, str(path))

    note = "UI included" if include_ui else "clean render without UI"
    sidecar = path.with_suffix(".txt")
    sidecar.write_text("\n".join(state_report_lines(state, sim_time, fps) + [f"capture_note: {note}"]) + "\n")
    return path


def write_benchmark_report(state, frame_ms_samples: list[float], sim_time: float) -> Path:
    safe_samples = [max(float(v), 0.001) for v in frame_ms_samples if v is not None]
    if not safe_samples:
        safe_samples = [0.001]

    avg_ms = sum(safe_samples) / len(safe_samples)
    min_ms = min(safe_samples)
    max_ms = max(safe_samples)
    avg_fps = 1000.0 / avg_ms
    max_fps = 1000.0 / min_ms
    min_fps = 1000.0 / max_ms

    stem = export_stem(state, "benchmark", include_ui=state.show_ui)
    report_path = export_dir() / f"{stem}.txt"
    csv_path = report_path.with_suffix(".csv")

    lines = [
        "Real-Time Black Hole Simulator benchmark",
        f"frames: {len(safe_samples)}",
        f"avg_frame_ms: {avg_ms:.3f}",
        f"min_frame_ms: {min_ms:.3f}",
        f"max_frame_ms: {max_ms:.3f}",
        f"avg_fps: {avg_fps:.3f}",
        f"min_fps: {min_fps:.3f}",
        f"max_fps: {max_fps:.3f}",
        "",
        *state_report_lines(state, sim_time, avg_fps),
    ]
    report_path.write_text("\n".join(lines) + "\n")

    csv_lines = ["frame,frame_ms,instant_fps"]
    for i, ms in enumerate(safe_samples, start=1):
        csv_lines.append(f"{i},{ms:.6f},{1000.0 / ms:.6f}")
    csv_path.write_text("\n".join(csv_lines) + "\n")
    return report_path


UI_TITLE = (238, 238, 238)
UI_TEXT = (230, 230, 232)
UI_MUTED = (185, 188, 194)
UI_PANEL_BG = (16, 16, 18, 178)
UI_PANEL_BORDER = (105, 108, 116, 86)
UI_LINE = (255, 255, 255, 18)
UI_SLIDER_TRACK = (54, 55, 60)
UI_SLIDER_FILL = (145, 147, 152)
UI_SLIDER_KNOB = (215, 216, 220)

LEFT_X = 12
RIGHT_X = 1304
PANEL_W = 220
SLIDER_X = RIGHT_X + 16
SLIDER_W = 172
CHECK_X = RIGHT_X + 16


def draw_panel(screen, rect, title=None):
    surf = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(surf, UI_PANEL_BG, surf.get_rect(), border_radius=8)
    pygame.draw.rect(surf, UI_PANEL_BORDER, surf.get_rect(), 1, border_radius=8)
    pygame.draw.line(surf, UI_LINE, (1, 33), (rect[2] - 2, 33), 1)
    screen.blit(surf, rect[:2])
    if title:
        title_font = pygame.font.SysFont("arial", 14, bold=True)
        draw_text(screen, title_font, title, rect[0] + 14, rect[1] + 14, UI_TITLE)


def draw_checkbox(screen, font, label, checked, x, y):
    box = pygame.Rect(x, y, 16, 16)
    pygame.draw.rect(screen, (215, 216, 220) if checked else (18, 18, 20), box, border_radius=3)
    pygame.draw.rect(screen, (232, 232, 235), box, 1, border_radius=3)
    if checked:
        pygame.draw.line(screen, (20, 20, 22), (x + 3, y + 8), (x + 7, y + 12), 2)
        pygame.draw.line(screen, (20, 20, 22), (x + 7, y + 12), (x + 14, y + 3), 2)
    draw_text(screen, font, label, x + 26, y - 1, UI_TEXT)


def draw_slider(screen, font, label, value, x, y, minv, maxv, width=172):
    draw_text(screen, font, label, x, y, UI_TEXT)
    draw_text(screen, font, f"{value:.2f}", x + width - 44, y, UI_TEXT)
    bar_y = y + 30
    pygame.draw.rect(screen, UI_SLIDER_TRACK, (x, bar_y, width, 6), border_radius=4)
    f = clamp((value - minv) / (maxv - minv), 0.0, 1.0)
    knob_x = int(x + f * width)
    pygame.draw.rect(screen, UI_SLIDER_FILL, (x, bar_y, max(2, knob_x - x), 6), border_radius=4)
    pygame.draw.circle(screen, UI_SLIDER_KNOB, (knob_x, bar_y + 3), 7)


def draw_button(screen, font, label, rect):
    pygame.draw.rect(screen, (47, 47, 50, 225), rect, border_radius=6)
    pygame.draw.rect(screen, (118, 120, 126, 90), rect, 1, border_radius=6)
    tx = rect[0] + rect[2] // 2 - font.size(label)[0] // 2
    ty = rect[1] + rect[3] // 2 - font.get_height() // 2
    draw_text(screen, font, label, tx, ty, UI_TEXT)


def draw_status_table(screen, font, x, y, rows, col2=108, line=26):
    for i, (k, v) in enumerate(rows):
        yy = y + i * line
        draw_text(screen, font, k, x, yy, UI_TEXT)
        draw_text(screen, font, v, x + col2, yy, UI_TEXT)


def draw_ui(screen, font, small_font, state, fps, sim_time):
    if not state.show_ui:
        return

    # Left-side panels show simulation state and camera controls in a compact form.
    draw_panel(screen, (LEFT_X, 52, 220, 178), "Simulation Info")
    frame_ms = 1000.0 / max(fps, 1.0)
    rows = [
        ("BH Mass:", f"{state.mass:.2f} M_sun"),
        ("Spin (a):", f"{state.spin:.2f}"),
        ("Accretion Rate:", f"{state.accretion:.2f}"),
        ("Time Scale:", f"{state.time_scale:.2f} x"),
        ("Frame Time:", f"{frame_ms:4.1f} ms ({fps:.0f} FPS)"),
    ]
    draw_status_table(screen, font, LEFT_X + 14, 96, rows, 104, 25)

    draw_panel(screen, (LEFT_X, 246, 220, 152), "Camera")
    rows = [
        ("Yaw:", f"{state.yaw * 180 / 3.14159:5.1f}°"),
        ("Pitch:", f"{state.pitch * 180 / 3.14159:5.1f}°"),
        ("Distance:", f"{state.distance:4.1f}"),
        ("FOV:", "28.2°"),
    ]
    draw_status_table(screen, font, LEFT_X + 14, 292, rows, 92, 25)

    draw_panel(screen, (LEFT_X, 412, 220, 438), "Controls")
    controls = [
        ("Mouse", "Orbit"),
        ("Scroll", "Zoom"),
        ("A / D", "Orbit L/R"),
        ("Q / E", "Orbit U/D"),
        ("W / S", "Near/Far"),
        ("V / F", "Spin +/-"),
        ("Z / X", "Mass -/+"),
        ("T / Y", "Accretion -/+"),
        ("N / M", "Science -/+"),
        ("J / K", "Stars -/+"),
        ("1", "Cinematic"),
        ("2", "Balanced"),
        ("3", "Strong physics"),
        ("4", "Diagnostic"),
        ("5 / 6", "Demo face/close"),
        ("7", "Explain physics"),
        ("8", "Cinematic export"),
        ("P", "Screenshot"),
        ("O", "Clean screenshot"),
        ("B", "Benchmark"),
        ("G", "Grid"),
        ("C", "Overlay"),
        ("H", "UI"),
        ("R", "Reset"),
        ("ESC", "Quit"),
    ]
    y = 454
    for k, v in controls:
        draw_text(screen, small_font, k, LEFT_X + 14, y, UI_TEXT)
        draw_text(screen, small_font, v, LEFT_X + 78, y, UI_TEXT)
        y += 15

    # Right-side panels hold visibility toggles, physical parameters, and presets.
    draw_panel(screen, (RIGHT_X, 52, PANEL_W, 242), "Visualization")
    items = [
        ("show_disk", "Accretion Disk", state.show_disk),
        ("show_ring", "Photon Ring", state.show_ring),
        ("show_grid", "Spacetime Grid", state.show_grid),
        ("show_stars", "Stars", state.show_stars),
        ("show_particles", "Infalling Particles", state.show_particles),
    ]
    for i, (_, label, val) in enumerate(items):
        draw_checkbox(screen, font, label, val, CHECK_X, 96 + i * 28)
    draw_text(screen, font, "Sky Mode", SLIDER_X, 242, UI_TEXT)
    pygame.draw.rect(screen, (38, 38, 40), (SLIDER_X, 264, SLIDER_W, 30), border_radius=5)
    pygame.draw.rect(screen, (105, 108, 116, 72), (SLIDER_X, 264, SLIDER_W, 30), 1, border_radius=5)
    draw_text(screen, font, "Procedural Stars", SLIDER_X + 10, 271, UI_TEXT)
    draw_text(screen, font, "v", SLIDER_X + SLIDER_W - 20, 270, UI_TEXT)

    draw_panel(screen, (RIGHT_X, 310, PANEL_W, 384), "Parameters")
    draw_slider(screen, font, "BH Mass (M_sun)", state.mass, SLIDER_X, 354, 4.0, 20.0, SLIDER_W)
    draw_slider(screen, font, "Spin (a)", state.spin, SLIDER_X, 406, 0.0, 1.0, SLIDER_W)
    draw_slider(screen, font, "Accretion Rate", state.accretion, SLIDER_X, 458, 0.0, 1.5, SLIDER_W)
    draw_slider(screen, font, "Time Scale", state.time_scale, SLIDER_X, 510, 0.1, 3.0, SLIDER_W)
    draw_slider(screen, font, "Science Strength", state.science_strength, SLIDER_X, 562, 0.0, 1.0, SLIDER_W)
    draw_slider(screen, font, "Star Detail", state.star_detail, SLIDER_X, 614, 0.0, 1.0, SLIDER_W)
    draw_button(screen, font, "Reset to Default", (SLIDER_X, 664, SLIDER_W, 30))

    draw_panel(screen, (RIGHT_X, 708, PANEL_W, 154), "Camera Presets")
    buttons = [
        ("Face On", 744),
        ("Edge On", 768),
        ("Top Down", 792),
        ("Close Up", 816),
        ("Far View", 840),
    ]
    for label, by in buttons:
        draw_button(screen, font, label, (SLIDER_X, by, SLIDER_W, 22))

    # Bottom readout keeps simulation time, science mode, star detail, and status visible.
    draw_text(screen, font, f"Time: {sim_time:5.1f} s", 14, HEIGHT - 28, UI_TEXT)
    draw_text(
        screen,
        font,
        f"Science: {state.science_strength:.2f} ({science_mode_label(state.science_strength)})",
        180,
        HEIGHT - 28,
        UI_MUTED,
    )
    draw_text(screen, font, f"Stars: {state.star_detail:.2f}", 390, HEIGHT - 28, UI_MUTED)
    if state.science_overlay:
        draw_text(
            screen,
            font,
            "Diagnostic overlay: ISCO / horizon / photon critical curve",
            510,
            HEIGHT - 28,
            UI_MUTED,
        )
    if state.status_message:
        draw_text(screen, small_font, state.status_message[:92], 510, HEIGHT - 50, UI_MUTED)

def reset(state):
    state.mass = 9.87
    state.spin = 0.85
    state.accretion = 0.82
    state.time_scale = 1.0
    state.yaw = 0.0
    state.pitch = 0.16
    state.distance = 5.8
    state.show_disk = True
    state.show_ring = True
    state.show_grid = True
    state.show_stars = True
    state.show_particles = True
    state.show_ui = True
    state.exposure = 1.30
    state.bloom = 1.55
    state.contrast = 1.06
    state.saturation = 0.94
    state.science_strength = 0.0
    state.star_detail = 0.68
    state.science_overlay = False


def set_science_preset(state, name):
    if name == "cinematic":
        state.science_strength = 0.0
        state.science_overlay = False
    elif name == "balanced":
        state.science_strength = 0.35
        state.science_overlay = False
    elif name == "science":
        state.science_strength = 0.75
        state.science_overlay = False
    elif name == "diagnostic":
        state.science_strength = 0.75
        state.science_overlay = True
        state.show_ui = True


def set_camera_preset(state, name):
    if name == "Face On":
        state.yaw = 0.0
        state.pitch = 0.16
        state.distance = 5.8
    elif name == "Edge On":
        state.yaw = 0.0
        state.pitch = 0.36
        state.distance = 5.9
    elif name == "Top Down":
        state.pitch = 1.12
        state.distance = 6.8
    elif name == "Close Up":
        state.pitch = 0.12
        state.distance = 3.8
    elif name == "Far View":
        state.pitch = 0.16
        state.distance = 7.6


def apply_public_demo_preset(state, camera_name: str = "Face On") -> None:
    # Public-demo preset: stable physical parameters, UI visible, and overlay hidden.
    reset(state)
    set_science_preset(state, "balanced")
    set_camera_preset(state, camera_name)
    state.show_ui = True
    state.star_detail = 0.72
    state.science_overlay = False


def apply_physics_explanation_preset(state) -> None:
    # Physics-explanation preset: diagnostic guides are visible and the UI stays on.
    reset(state)
    set_science_preset(state, "diagnostic")
    set_camera_preset(state, "Face On")
    state.show_ui = True
    state.star_detail = 0.68


def apply_cinematic_output_preset(state) -> None:
    # Cinematic-output preset: hide the UI, keep particles active, and preserve the
    # user's current grid choice.
    keep_grid = state.show_grid
    reset(state)
    set_science_preset(state, "balanced")
    set_camera_preset(state, "Close Up")
    state.show_ui = False
    state.science_overlay = False
    state.show_particles = True
    state.star_detail = 0.78
    state.show_grid = keep_grid


def set_slider(state, name, mx):
    table = {
        "mass": (SLIDER_X, SLIDER_W, 4.0, 20.0),
        "spin": (SLIDER_X, SLIDER_W, 0.0, 1.0),
        "accretion": (SLIDER_X, SLIDER_W, 0.0, 1.5),
        "time_scale": (SLIDER_X, SLIDER_W, 0.1, 3.0),
        "science_strength": (SLIDER_X, SLIDER_W, 0.0, 1.0),
        "star_detail": (SLIDER_X, SLIDER_W, 0.0, 1.0),
    }

    sx, sw, mn, mxv = table[name]
    f = clamp((mx - sx) / sw, 0.0, 1.0)
    setattr(state, name, mn + f * (mxv - mn))


def handle_click(state, pos):
    x, y = pos

    checks = [
        ("show_disk", CHECK_X, 96),
        ("show_ring", CHECK_X, 124),
        ("show_grid", CHECK_X, 152),
        ("show_stars", CHECK_X, 180),
        ("show_particles", CHECK_X, 208),
    ]

    for name, cx, cy in checks:
        if cx <= x <= cx + 180 and cy - 4 <= y <= cy + 22:
            setattr(state, name, not getattr(state, name))
            return

    sliders = [
        ("mass", SLIDER_X, 384),
        ("spin", SLIDER_X, 436),
        ("accretion", SLIDER_X, 488),
        ("time_scale", SLIDER_X, 540),
        ("science_strength", SLIDER_X, 592),
        ("star_detail", SLIDER_X, 644),
    ]

    for name, sx, sy in sliders:
        if sx <= x <= sx + SLIDER_W and sy - 10 <= y <= sy + 15:
            state.dragging_slider = name
            set_slider(state, name, x)
            return

    if SLIDER_X <= x <= SLIDER_X + SLIDER_W and 664 <= y <= 694:
        reset(state)
        return

    presets = [
        ("Face On", 744),
        ("Edge On", 768),
        ("Top Down", 792),
        ("Close Up", 816),
        ("Far View", 840),
    ]
    for name, by in presets:
        if SLIDER_X <= x <= SLIDER_X + SLIDER_W and by <= y <= by + 22:
            set_camera_preset(state, name)
            return

    # Ignore orbit dragging when the mouse is over the side-control panels.
    if x < 260 or x > 1284:
        return
    state.dragging = True

def main():
    pygame.init()

    # The sky is generated procedurally in the shader, so startup does not depend
    # on loading a large external panorama.
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK,
        pygame.GL_CONTEXT_PROFILE_CORE,
    )

    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Real-Time Black Hole Simulator - Star-Only Repo Version")

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
    font = pygame.font.SysFont("arial", 14)
    small_font = pygame.font.SysFont("arial", 13)

    t = 0.0
    running = True
    last_mouse = None
    pending_screenshot = None
    benchmark_active = False
    benchmark_samples = []

    while running:
        frame_ms_raw = clock.tick(FPS)
        dt = frame_ms_raw / 60.0
        fps = clock.get_fps()
        t += dt * state.time_scale

        now_ms = pygame.time.get_ticks()
        if state.status_message and state.status_until_ms and now_ms > state.status_until_ms:
            state.status_message = ""
            state.status_until_ms = 0

        if benchmark_active:
            benchmark_samples.append(max(float(frame_ms_raw), 0.001))
            if len(benchmark_samples) % 15 == 0:
                set_status(state, f"Benchmarking current mode: {len(benchmark_samples)}/{BENCHMARK_FRAMES} frames", 1.5)
            if len(benchmark_samples) >= BENCHMARK_FRAMES:
                report_path = write_benchmark_report(state, benchmark_samples, t)
                benchmark_active = False
                benchmark_samples = []
                msg = f"Benchmark saved: {report_path.name}"
                print(msg)
                set_status(state, msg, 5.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:
                    state.show_grid = not state.show_grid
                elif event.key == pygame.K_c:
                    state.science_overlay = not state.science_overlay
                elif event.key == pygame.K_h:
                    state.show_ui = not state.show_ui
                elif event.key == pygame.K_r:
                    reset(state)
                    set_status(state, "Reset to default cinematic baseline")
                elif event.key == pygame.K_1:
                    set_science_preset(state, "cinematic")
                    set_status(state, "Mode 1: cinematic baseline")
                elif event.key == pygame.K_2:
                    set_science_preset(state, "balanced")
                    set_status(state, "Mode 2: balanced science")
                elif event.key == pygame.K_3:
                    set_science_preset(state, "science")
                    set_status(state, "Mode 3: strong physics, overlay off")
                elif event.key == pygame.K_4:
                    set_science_preset(state, "diagnostic")
                    set_status(state, "Mode 4: diagnostic overlay, UI on")
                elif event.key == pygame.K_5:
                    apply_public_demo_preset(state, "Face On")
                    set_status(state, "Public demo preset: face-on balanced science")
                elif event.key == pygame.K_6:
                    apply_public_demo_preset(state, "Close Up")
                    set_status(state, "Public demo preset: close-up balanced science")
                elif event.key == pygame.K_7:
                    apply_physics_explanation_preset(state)
                    set_status(state, "Physics explanation preset: diagnostic overlay on")
                elif event.key == pygame.K_8:
                    apply_cinematic_output_preset(state)
                    msg = "Cinematic export preset: UI hidden, balanced science"
                    print(msg)
                    set_status(state, msg)
                elif event.key == pygame.K_p:
                    pending_screenshot = "with_ui"
                    set_status(state, "Saving screenshot after this frame...", 1.5)
                elif event.key == pygame.K_o:
                    pending_screenshot = "no_ui"
                    set_status(state, "Saving clean screenshot without UI...", 1.5)
                elif event.key == pygame.K_b:
                    benchmark_active = True
                    benchmark_samples = []
                    msg = f"Benchmarking current mode for {BENCHMARK_FRAMES} frames..."
                    print(msg)
                    set_status(state, msg, 2.5)

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
        if keys[pygame.K_v]:
            state.spin = clamp(state.spin + 0.008, 0.0, 1.0)
        if keys[pygame.K_f]:
            state.spin = clamp(state.spin - 0.008, 0.0, 1.0)
        if keys[pygame.K_z]:
            state.mass = clamp(state.mass - 0.06, 4.0, 20.0)
        if keys[pygame.K_x]:
            state.mass = clamp(state.mass + 0.06, 4.0, 20.0)
        if keys[pygame.K_t]:
            state.accretion = clamp(state.accretion - 0.006, 0.0, 1.5)
        if keys[pygame.K_y]:
            state.accretion = clamp(state.accretion + 0.006, 0.0, 1.5)
        if keys[pygame.K_n]:
            state.science_strength = clamp(state.science_strength - 0.008, 0.0, 1.0)
        if keys[pygame.K_m]:
            state.science_strength = clamp(state.science_strength + 0.008, 0.0, 1.0)
        if keys[pygame.K_j]:
            state.star_detail = clamp(state.star_detail - 0.010, 0.0, 1.0)
        if keys[pygame.K_k]:
            state.star_detail = clamp(state.star_detail + 0.010, 0.0, 1.0)

        program["u_resolution"].value = (WIDTH, HEIGHT)
        program["u_time"].value = t
        program["u_mass"].value = state.mass
        program["u_spin"].value = state.spin
        program["u_isco"].value = kerr_isco_radius(state.spin)
        program["u_horizon_kerr"].value = kerr_horizon_radius(state.spin)
        program["u_accretion"].value = state.accretion
        program["u_yaw"].value = state.yaw
        program["u_pitch"].value = state.pitch
        program["u_distance"].value = state.distance
        program["u_show_disk"].value = 1.0 if state.show_disk else 0.0
        program["u_show_ring"].value = 1.0 if state.show_ring else 0.0
        program["u_show_grid"].value = 1.0 if state.show_grid else 0.0
        program["u_show_stars"].value = 1.0 if state.show_stars else 0.0
        program["u_show_particles"].value = 1.0 if state.show_particles else 0.0
        program["u_exposure"].value = state.exposure
        program["u_contrast"].value = state.contrast
        program["u_saturation"].value = state.saturation
        program["u_bloom"].value = state.bloom
        program["u_science_overlay"].value = 1.0 if state.science_overlay else 0.0
        program["u_science_strength"].value = state.science_strength
        program["u_star_detail"].value = state.star_detail
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render()

        if pending_screenshot == "no_ui":
            screenshot_path = save_framebuffer(ctx, state, t, fps, include_ui=False)
            msg = f"Saved clean screenshot: {screenshot_path.name}"
            print(msg)
            set_status(state, msg, 5.0)
            pending_screenshot = None

        if state.show_ui:
            ui_surface.fill((0, 0, 0, 0))
            draw_ui(ui_surface, font, small_font, state, fps, t)

            ui_data = pygame.image.tostring(ui_surface, "RGBA", True)
            ui_texture.write(ui_data)
            ui_texture.use(0)
            ui_program["u_texture"].value = 0

            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            ui_vao.render()
            ctx.disable(moderngl.BLEND)

        if pending_screenshot == "with_ui":
            screenshot_path = save_framebuffer(ctx, state, t, fps, include_ui=state.show_ui)
            label = "screenshot" if state.show_ui else "screenshot without UI"
            msg = f"Saved {label}: {screenshot_path.name}"
            print(msg)
            set_status(state, msg, 5.0)
            pending_screenshot = None

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
