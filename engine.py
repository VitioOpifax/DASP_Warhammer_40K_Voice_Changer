import numpy as np
import scipy.signal as signal

# =====================================================================
# WEEK 1-2: FUNDAMENTALS
# =====================================================================

def bitcrusher(x, bit_depth=8.0):
    """Simulates ADC quantization degradation (Maybe used for Vox-caster static)."""
    bits = float(bit_depth)
    step = 2.0 ** bits
    return np.round(x * step) / step

def resampling_pitch_shift(x, pitch_factor=0.94):
    """Alters the playback rate via array resampling (Change Physical weight)."""
    factor = float(pitch_factor)
    new_length = int(len(x) / factor)
    original_positions = np.arange(len(x))
    new_positions = np.linspace(0, len(x) - 1, new_length)
    return np.interp(new_positions, original_positions, x)


# =====================================================================
# WEEKS 5-6: FILTERS
# =====================================================================

def low_pass_filter(x, fs, cutoff=3000.0, order=4):
    """Removes high frequencies (muffles sound, simulates thick armor)."""
    sos = signal.butter(int(order), float(cutoff), btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, x)

def high_pass_filter(x, fs, cutoff=300.0, order=4):
    """Removes low frequencies (thins out sound, removes rumble)."""
    sos = signal.butter(int(order), float(cutoff), btype='high', fs=fs, output='sos')
    return signal.sosfilt(sos, x)

def band_pass_filter(x, fs, low_cut=300.0, high_cut=3000.0, order=4):
    """Isolates a specific frequency range."""
    sos = signal.butter(int(order), [float(low_cut), float(high_cut)], btype='band', fs=fs, output='sos')
    return signal.sosfilt(sos, x)

def notch_filter(x, fs, center_freq=1000.0, q_factor=30.0):
    """Cuts a specific frequency (Could be used as a static De-Esser)."""
    b, a = signal.iirnotch(float(center_freq), float(q_factor), fs=fs)
    return signal.lfilter(b, a, x)


# =====================================================================
# WEEK 7-9: TIME-DOMAIN EFFECTS
# =====================================================================

def pure_delay(x, fs, delay_ms=50.0):
    """A clean, single repetition of the signal."""
    delay_samples = int((float(delay_ms) / 1000.0) * fs)
    y = np.zeros_like(x)
    y[delay_samples:] = x[:-delay_samples]
    return y

def feedback_delay(x, fs, delay_ms=30.0, feedback=0.6):
    """Decaying repetitions (Sound reflections)."""
    D = int((float(delay_ms) / 1000.0) * fs)
    y = np.zeros_like(x)
    fb_val = float(feedback)

    for n in range(len(x)):
        if n < D:
            y[n] = x[n]
        else:
            y[n] = x[n] + fb_val * y[n - D]
    return y

def synthetic_reverb(x, fs, rt60=1.2, mix=0.3): # More advanced.
    """Convolution reverb using a synthetically generated room impulse response."""
    # Generate an exponentially decaying white noise Impulse Response
    np.random.seed(42) # Locks the RNG
    t = np.arange(int(fs * float(rt60))) / fs
    ir = np.random.randn(len(t)) * np.exp(-t * (6.91 / float(rt60)))

    # Bandpass the IR to make it sound like a metallic hangar
    sos = signal.butter(2, [500.0, 4000.0], btype='band', fs=fs, output='sos')
    ir = signal.sosfilt(sos, ir)
    ir = ir / np.max(np.abs(ir)) # Normalize IR

    y = signal.fftconvolve(x, ir, mode='full')[:len(x)]
    m = float(mix)
    return (1.0 - m) * x + m * y


# =====================================================================
# WEEK 11: MODULATION
# =====================================================================

def tremolo(x, fs, rate=5.0, depth=0.5):
    """LFO applied to the volume amplitude."""
    t = np.arange(len(x)) / fs
    lfo = 1.0 - float(depth) + float(depth) * (0.5 * (1.0 + np.sin(2 * np.pi * float(rate) * t)))
    return x * lfo

def chorus(x, fs, rate=1.5, depth_ms=2.0, base_delay_ms=15.0, mix=0.5):
    """Vectorized feedforward fractional delay line (Thickens voice)."""
    t = np.arange(len(x)) / fs
    delay_sec = (float(base_delay_ms) / 1000.0) + (float(depth_ms) / 1000.0) * np.sin(2 * np.pi * float(rate) * t)
    delay_samples = delay_sec * fs

    indices = np.arange(len(x)) - delay_samples
    valid = indices >= 0
    indices_floor = np.floor(indices).astype(int)
    indices_ceil = indices_floor + 1
    frac = indices - indices_floor

    # Clip safely
    indices_floor = np.clip(indices_floor, 0, len(x) - 1)
    indices_ceil = np.clip(indices_ceil, 0, len(x) - 1)

    y = np.zeros_like(x)
    y[valid] = x[indices_floor[valid]] * (1 - frac[valid]) + x[indices_ceil[valid]] * frac[valid]

    m = float(mix)
    return (1.0 - m) * x + m * y

def ring_modulator(x, fs, mod_freq=50.0, mix=1.0):
    """Multiplies signal by an audio-rate oscillator (robotic tone)."""
    t = np.arange(len(x)) / fs
    rm = x * np.sin(2 * np.pi * float(mod_freq) * t)
    m = float(mix)
    return (1.0 - m) * x + m * rm


# =====================================================================
# THE BLOCK REGISTRY
# =====================================================================

BLOCK_LIBRARY = {
    "Bitcrusher": {"func": bitcrusher, "needs_fs": False, "default_params": {"bit_depth": 8.0}},
    "Pitch Shift": {"func": resampling_pitch_shift, "needs_fs": False, "default_params": {"pitch_factor": 0.90}},
    "Low-Pass Filter": {"func": low_pass_filter, "needs_fs": True, "default_params": {"cutoff": 3000.0, "order": 4}},
    "High-Pass Filter": {"func": high_pass_filter, "needs_fs": True, "default_params": {"cutoff": 300.0, "order": 4}},
    "Band-Pass Filter": {"func": band_pass_filter, "needs_fs": True, "default_params": {"low_cut": 300.0, "high_cut": 3000.0, "order": 4}},
    "Notch Filter": {"func": notch_filter, "needs_fs": True, "default_params": {"center_freq": 6000.0, "q_factor": 10.0}},
    "Delay": {"func": pure_delay, "needs_fs": True, "default_params": {"delay_ms": 50.0}},
    "Feedback Delay": {"func": feedback_delay, "needs_fs": True, "default_params": {"delay_ms": 30.0, "feedback": 0.6}},
    "Synthetic Reverb": {"func": synthetic_reverb, "needs_fs": True, "default_params": {"rt60": 1.2, "mix": 0.3}},
    "Tremolo": {"func": tremolo, "needs_fs": True, "default_params": {"rate": 5.0, "depth": 0.5}},
    "Chorus": {"func": chorus, "needs_fs": True, "default_params": {"rate": 1.5, "depth_ms": 2.0, "base_delay_ms": 15.0, "mix": 0.5}},
    "Ring Modulator": {"func": ring_modulator, "needs_fs": True, "default_params": {"mod_freq": 55.0, "mix": 0.8}}
}

def get_available_blocks():
    """Returns a dictionary of available blocks and their default parameters for the GUI."""
    return {name: info["default_params"] for name, info in BLOCK_LIBRARY.items()}


# =====================================================================
# PIPELINE PROCESSOR
# =====================================================================

def process_audio(x, fs, profile_data):
    """
    Takes an audio array and dynamically routes it through the blocks
    defined in the JSON profile.
    """
    y = x.copy()

    # Force mono channel for uniform processing
    if y.ndim > 1:
        y = y[:, 0]

    pipeline = profile_data.get("pipeline", [])

    for block in pipeline:
        b_name = block.get("block_name")
        b_params = block.get("parameters", {})

        if b_name in BLOCK_LIBRARY:
            func = BLOCK_LIBRARY[b_name]["func"]
            needs_fs = BLOCK_LIBRARY[b_name]["needs_fs"]

            print(f" -> Applying {b_name} with parameters: {b_params}")
            if needs_fs:
                y = func(y, fs, **b_params)
            else:
                y = func(y, **b_params)
        else:
            print(f"Warning: Block '{b_name}' not found in the Engine registry.")

    # Postprocessing: Normalize audio levels to prevent clipping
    max_amplitude = np.max(np.abs(y))
    if max_amplitude > 0:
        y = y / max_amplitude * 0.95  # Normalize to -0.44 dBFS

    return y
