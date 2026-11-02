
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


def extract_audio_features(audio_path):

    # Load original audio
    audio, sr = librosa.load(
        audio_path,
        sr=8000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # 1. Pitch extraction
    pitch = sound.to_pitch(
        time_step=0.01,
        pitch_floor=75,
        pitch_ceiling=500
    )

    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) == 0:
        raise ValueError("No voiced pitch detected")

    # 2. Intensity
    intensity = sound.to_intensity(
        minimum_pitch=75,
        time_step=0.01
    )

    intensity_values = intensity.values.flatten()
    intensity_values = intensity_values[
        np.isfinite(intensity_values)
    ]

    # 3. Harmonicity / HNR
    harmonicity = sound.to_harmonicity_cc(
        time_step=0.01,
        minimum_pitch=75
    )

    hnr_values = harmonicity.values.flatten()
    hnr_values = hnr_values[np.isfinite(hnr_values)]

    # 4. Jitter and shimmer
    point_process = call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        500
    )

    try:
        jitter = call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )
    except Exception:
        jitter = np.nan

    try:
        shimmer = call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        shimmer = np.nan

    # 5. MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=128
    )

    # 6. RMS features
    rms = librosa.feature.rms(
        y=audio,
        frame_length=512,
        hop_length=128
    ).flatten()

    # 7. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=512,
        hop_length=128
    ).flatten()

    features = {
        "pitch_mean": np.mean(pitch_values),
        "pitch_std": np.std(pitch_values),
        "pitch_min": np.min(pitch_values),
        "pitch_max": np.max(pitch_values),

        "intensity_mean": np.mean(intensity_values),
        "intensity_std": np.std(intensity_values),

        "hnr_mean": np.mean(hnr_values) if len(hnr_values) else np.nan,

        "jitter_local": jitter,
        "shimmer_local": shimmer,

        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),

        "zcr_mean": np.mean(zcr),
        "zcr_std": np.std(zcr),

        "duration": len(audio) / sr
    }

    # MFCC mean and standard deviation
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    return features
