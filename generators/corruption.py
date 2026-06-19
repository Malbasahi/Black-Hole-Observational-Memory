"""
corruption.py
-------------
Simulates observation degradation for synthetic black hole images.

This version avoids pixel dropout and uses Fourier / uv-plane corruption,
which is a better approximation for radio interferometry.

Stages:
    A. Atmospheric blur
    B. Thermal Gaussian noise
    C. uv-plane sparse sampling
    D. Fourier phase / amplitude distortion

This is still a simplified model, not a full VLBI simulator.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clip01(image: np.ndarray) -> np.ndarray:
    """Clip image to [0, 1] and return float32."""
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _fft2_centered(image: np.ndarray) -> np.ndarray:
    """Centered 2D FFT."""
    return np.fft.fftshift(np.fft.fft2(image.astype(np.float32)))


def _ifft2_centered(freq: np.ndarray) -> np.ndarray:
    """Inverse centered 2D FFT."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq))).astype(np.float32)


def _frequency_radius(shape: Tuple[int, int]) -> np.ndarray:
    """Return normalized centered frequency radius grid."""
    h, w = shape

    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]

    freq_r = np.sqrt(fx**2 + fy**2)
    freq_r = freq_r / (freq_r.max() + 1e-8)

    return freq_r.astype(np.float32)


# ---------------------------------------------------------------------------
# A. Gaussian noise
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Additive Gaussian thermal noise.

    Parameters
    ----------
    image:
        Input image in [0, 1].
    sigma:
        Noise standard deviation.
    rng:
        NumPy random generator.

    Returns
    -------
    Noisy image in [0, 1].
    """
    if sigma <= 0.0:
        return image.astype(np.float32)

    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return _clip01(image + noise)


# ---------------------------------------------------------------------------
# B. Atmospheric / optical blur
# ---------------------------------------------------------------------------

def add_atmospheric_blur(
    image: np.ndarray,
    blur_sigma: float,
) -> np.ndarray:
    """
    Apply additional observational blur.

    This is separate from the clean-image blur used in the generator.
    """
    if blur_sigma <= 0.0:
        return image.astype(np.float32)

    blurred = gaussian_filter(image.astype(np.float32), sigma=blur_sigma)
    return _clip01(blurred)


# ---------------------------------------------------------------------------
# C. uv-plane sparse sampling
# ---------------------------------------------------------------------------

def make_uv_mask(
    shape: Tuple[int, int],
    keep_fraction: float,
    rng: np.random.Generator,
    low_freq_keep_radius: float = 0.08,
) -> np.ndarray:
    """
    Create a simplified uv-plane sampling mask.

    Low frequencies are always kept because they preserve global structure.
    Higher frequencies are randomly sampled.

    Parameters
    ----------
    shape:
        Image shape.
    keep_fraction:
        Fraction of high-frequency coefficients to keep.
    rng:
        NumPy random generator.
    low_freq_keep_radius:
        Central uv radius always retained.

    Returns
    -------
    Binary mask with values 0 or 1.
    """
    keep_fraction = float(np.clip(keep_fraction, 0.02, 1.0))

    freq_r = _frequency_radius(shape)

    random_mask = rng.random(shape).astype(np.float32) < keep_fraction
    low_freq_mask = freq_r <= low_freq_keep_radius

    mask = random_mask | low_freq_mask

    # Enforce conjugate symmetry approximately for real-valued output.
    mask = mask | np.flip(np.flip(mask, axis=0), axis=1)

    return mask.astype(np.float32)


def add_uv_sparse_sampling(
    image: np.ndarray,
    keep_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate incomplete interferometric uv coverage.

    This replaces random pixel dropout. The image is transformed to Fourier
    space, masked, then transformed back.
    """
    if keep_fraction >= 1.0:
        return image.astype(np.float32)

    freq = _fft2_centered(image)
    mask = make_uv_mask(image.shape, keep_fraction=keep_fraction, rng=rng)

    sampled_freq = freq * mask
    sampled = _ifft2_centered(sampled_freq)

    sampled = sampled - sampled.min()
    sampled = sampled / (sampled.max() + 1e-8)

    return _clip01(sampled)


# ---------------------------------------------------------------------------
# D. Fourier-space phase / amplitude distortion
# ---------------------------------------------------------------------------

def add_frequency_distortion(
    image: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add simplified Fourier-domain artefacts.

    Effects:
        - phase perturbation
        - high-frequency attenuation
        - mild ringing / dirty-beam-like residuals
    """
    if strength <= 0.0:
        return image.astype(np.float32)

    strength = float(np.clip(strength, 0.0, 1.0))

    freq = _fft2_centered(image)
    freq_r = _frequency_radius(image.shape)

    phase_noise = rng.normal(
        loc=0.0,
        scale=np.pi * 0.35 * strength,
        size=image.shape,
    ).astype(np.float32)

    phase_factor = np.exp(1j * phase_noise)

    cutoff = rng.uniform(0.35, 0.75)
    attenuation = np.exp(-((freq_r / cutoff) ** 2) * strength)

    distorted_freq = freq * phase_factor * attenuation

    distorted = _ifft2_centered(distorted_freq)

    distorted = distorted - distorted.min()
    distorted = distorted / (distorted.max() + 1e-8)

    alpha = np.clip(0.55 * strength, 0.0, 1.0)
    blended = (1.0 - alpha) * image + alpha * distorted

    return _clip01(blended)


# ---------------------------------------------------------------------------
# Master corruption pipeline
# ---------------------------------------------------------------------------

def corrupt_image(
    clean: np.ndarray,
    noise_level: float,
    blur_strength: float,
    rng: np.random.Generator,
    use_sparse: bool = True,
    use_fourier: bool = True,
) -> np.ndarray:
    """
    Apply the full observation corruption stack.

    Stage order:
        clean image
        -> observational blur
        -> uv sparse sampling
        -> Fourier phase/amplitude distortion
        -> Gaussian thermal noise

    Parameters
    ----------
    clean:
        Clean image in [0, 1].
    noise_level:
        Primary corruption strength, expected in [0, 0.30].
    blur_strength:
        Clean-generator blur parameter. Used here only as a mild extra blur.
    rng:
        NumPy random generator.
    use_sparse:
        Toggle uv sparse sampling.
    use_fourier:
        Toggle Fourier distortion.

    Returns
    -------
    Corrupted observation image in [0, 1].
    """
    img = _clip01(clean)

    noise_level = float(np.clip(noise_level, 0.0, 0.30))
    blur_strength = float(max(0.0, blur_strength))

    # A. Observational blur
    atm_sigma = 0.20 * blur_strength
    img = add_atmospheric_blur(img, atm_sigma)

    # B. uv sparse sampling
    if use_sparse:
        # noise_level 0.00 -> keep almost everything
        # noise_level 0.30 -> keep around 32 percent of high-frequency samples
        keep_fraction = 1.0 - 2.25 * noise_level
        keep_fraction = float(np.clip(keep_fraction, 0.32, 1.0))
        img = add_uv_sparse_sampling(img, keep_fraction=keep_fraction, rng=rng)

    # C. Fourier artefacts
    if use_fourier:
        fourier_strength = noise_level * 1.4
        img = add_frequency_distortion(img, strength=fourier_strength, rng=rng)

    # D. Thermal noise last
    # Keep this conservative; otherwise samples become visually unrecoverable.
    thermal_sigma = noise_level * 0.45
    img = add_gaussian_noise(img, thermal_sigma, rng)

    return _clip01(img)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from PIL import Image

    rng = np.random.default_rng(42)

    size = 128
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    r = np.sqrt(x**2 + y**2)

    clean = np.exp(-((r - 0.25) ** 2) / 0.003).astype(np.float32)
    clean[r < 0.13] = 0.0
    clean = _clip01(clean)

    corrupted = corrupt_image(
        clean=clean,
        noise_level=0.25,
        blur_strength=2.0,
        rng=rng,
    )

    Image.fromarray((clean * 255).astype(np.uint8), mode="L").save(
        "corruption_clean_test.png"
    )
    Image.fromarray((corrupted * 255).astype(np.uint8), mode="L").save(
        "corruption_noisy_test.png"
    )

    print("Saved corruption_clean_test.png and corruption_noisy_test.png")