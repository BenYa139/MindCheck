"""
mindcheck_risk.py — trained-model risk scoring for MindCheck
==============================================================
Drop this file next to mindcheck_app.py, and put mindcheck_model.pkl
alongside it. Add to requirements.txt:

    scikit-learn
    scipy

WHY POOLED AUDIO
----------------
The model was trained on ~65-second recordings. Testing on truncated
clips showed the signal collapses on very short audio:

    5 s   AUC 0.568   (too short)
    10 s  AUC 0.696
    20 s  AUC 0.716
    65 s  AUC 0.698

So this module concatenates every answer from the session into one
audio stream before extracting features. It refuses to score sessions
with under 10 seconds of pooled speech.

HONEST PERFORMANCE (5-fold CV, 100 speakers, one recording each)
    accuracy  64%
    AUC       0.698
    at threshold 0.39:  catches 41/50 cases, misses 9, 26/50 false alarms
    permutation test p = 0.006 (better than chance)
"""

import os
import pickle
import wave
import tempfile
import numpy as np
from scipy.fftpack import dct

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mindcheck_model.pkl")
MIN_AUDIO_SEC = 10.0

# --- must match training exactly ---
FRAME_MS = 30
MIN_PAUSE_SEC, MAX_PAUSE_SEC = 0.15, 2.5
NOISE_PCT, NOISE_MULT = 10, 2.5
N_MFCC, N_FILTERS, N_FFT = 13, 26, 512
F0_MIN, F0_MAX = 60, 400

_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        with open(MODEL_PATH, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


# ──────────────────────────────────────────────
# audio helpers
# ──────────────────────────────────────────────
def _read_wav(path):
    with wave.open(path, "rb") as wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if sw == 1:
        x -= 128.0
    if nch > 1:
        u = (x.size // nch) * nch
        x = x[:u].reshape(-1, nch).mean(axis=1)
    return x, sr


def pool_audio(audio_bytes_list):
    """Concatenate all session recordings into one normalised stream."""
    chunks, sr = [], None
    for b in audio_bytes_list:
        if not b:
            continue
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(b)
                path = f.name
            x, s = _read_wav(path)
            if x.size == 0:
                continue
            peak = np.max(np.abs(x))
            if peak > 0:
                x = x / peak          # normalise each clip before joining
            if sr is None:
                sr = s
            elif s != sr:
                continue              # skip mismatched sample rates
            chunks.append(x)
        except Exception:
            continue
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    if not chunks or sr is None:
        return None, None
    return np.concatenate(chunks), sr


def _runs(mask, value):
    out, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def _f0(frame, sr):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-6:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    if hi >= len(corr) or corr[0] <= 0:
        return 0.0
    seg = corr[lo:hi]
    if seg.size == 0:
        return 0.0
    k = int(np.argmax(seg))
    if seg[k] / corr[0] < 0.3:
        return 0.0
    return sr / (lo + k)


def _melbank(sr, n_fft, n_filters):
    h2m = lambda f: 2595 * np.log10(1 + f / 700.0)
    m2h = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = np.linspace(h2m(0), h2m(sr / 2), n_filters + 2)
    b = np.floor((n_fft + 1) * m2h(pts) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        l, c, r = b[m - 1], b[m], b[m + 1]
        c, r = max(c, l + 1), max(r, max(c, l + 1) + 1)
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / (r - c)
    return fb


def extract_features(x, sr):
    """Return the 49-feature dict, matching training exactly."""
    total = len(x) / sr
    if total < 2:
        return None
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    fl = int(sr * FRAME_MS / 1000)
    n = len(x) // fl
    if n < 10:
        return None
    fr = x[:n * fl].reshape(n, fl)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))

    floor = np.percentile(rms, NOISE_PCT)
    if floor <= 0:
        p = rms[rms > 0]
        floor = p.min() if p.size else 1e-9
    thr = floor * NOISE_MULT
    if (rms >= thr).sum() < n * 0.02:
        thr = float(rms.max()) * 0.08
    voiced = rms >= thr
    if voiced.sum() == 0:
        return None

    fs = FRAME_MS / 1000
    minp, maxp = max(int(MIN_PAUSE_SEC / fs), 1), int(MAX_PAUSE_SEC / fs)
    pauses, dead = [], 0
    for s0, ln in _runs(voiced, False):
        if s0 == 0 or s0 + ln == n or ln > maxp:
            dead += ln
        elif ln >= minp:
            pauses.append(ln * fs)

    af = n - dead
    if af <= 0:
        return None
    vf = int(voiced.sum())
    speak, analysed = vf * fs, af * fs
    sil = analysed - speak

    vruns = [l * fs for _, l in _runs(voiced, True)]
    vruns = [v for v in vruns if v >= 0.1]

    f0s = np.array([f for f in (_f0(fr[i], sr) for i in np.nonzero(voiced)[0])
                    if F0_MIN < f < F0_MAX])
    vrms = rms[voiced]

    xe = np.append(x[0], x[1:] - 0.97 * x[:-1])
    fl2, hop = int(sr * 25 / 1000), int(sr * 10 / 1000)
    nfr = 1 + (len(xe) - fl2) // hop
    idx = (np.tile(np.arange(fl2), (nfr, 1))
           + np.tile(np.arange(0, nfr * hop, hop), (fl2, 1)).T)
    f2 = xe[idx.astype(np.int32)] * np.hamming(fl2)
    pw = (1.0 / N_FFT) * np.abs(np.fft.rfft(f2, N_FFT)) ** 2
    en = np.dot(pw, _melbank(sr, N_FFT, N_FILTERS).T)
    en = np.where(en == 0, np.finfo(float).eps, en)
    mfcc = dct(np.log(en), type=2, axis=1, norm="ortho")[:, :N_MFCC]

    out = {
        "total_duration_sec": round(total, 2),
        "analysed_duration_sec": round(analysed, 2),
        "speaking_duration_sec": round(speak, 2),
        "silence_duration_sec": round(sil, 2),
        "dead_air_excluded_sec": round(dead * fs, 2),
        "silence_to_speech_ratio": round(sil / speak, 4) if speak > 0 else 0,
        "silence_ratio": round(sil / analysed, 4) if analysed > 0 else 0,
        "pause_count": len(pauses),
        "pause_rate_per_min": round(len(pauses) / (analysed / 60), 2) if analysed > 0 else 0,
        "avg_pause_duration_sec": round(float(np.mean(pauses)), 3) if pauses else 0.0,
        "longest_pause_sec": round(float(np.max(pauses)), 3) if pauses else 0.0,
        "total_pause_time_sec": round(float(np.sum(pauses)), 2) if pauses else 0.0,
        "pause_time_ratio": round(float(np.sum(pauses)) / analysed, 4) if pauses and analysed else 0,
        "phrase_count": len(vruns),
        "avg_phrase_length_sec": round(float(np.mean(vruns)), 3) if vruns else 0.0,
        "longest_phrase_sec": round(float(np.max(vruns)), 3) if vruns else 0.0,
        "phrase_length_std": round(float(np.std(vruns)), 3) if vruns else 0.0,
        "f0_mean_hz": round(float(np.mean(f0s)), 2) if f0s.size else 0.0,
        "f0_std_hz": round(float(np.std(f0s)), 2) if f0s.size else 0.0,
        "f0_range_hz": round(float(np.percentile(f0s, 95) - np.percentile(f0s, 5)), 2) if f0s.size > 10 else 0.0,
        "intensity_mean": round(float(np.mean(vrms)), 4),
        "intensity_std": round(float(np.std(vrms)), 4),
        "intensity_cv": round(float(np.std(vrms) / np.mean(vrms)), 4) if np.mean(vrms) > 0 else 0,
    }
    for i in range(N_MFCC):
        out[f"mfcc{i+1}_mean"] = round(float(np.mean(mfcc[:, i])), 4)
        out[f"mfcc{i+1}_std"] = round(float(np.std(mfcc[:, i])), 4)
    return out


# ──────────────────────────────────────────────
# main entry point
# ──────────────────────────────────────────────
def score_session(audio_bytes_list):
    """
    Run the trained model on all recordings from one session.

    Returns dict:
        status      "ok" | "insufficient_audio" | "error"
        probability model output, 0-1
        flagged     True if probability >= threshold
        band        "typical" | "borderline" | "above_typical"
        speech_sec  pooled speech analysed
        features    the 49 extracted features
    """
    x, sr = pool_audio(audio_bytes_list)
    if x is None:
        return {"status": "error", "message": "No usable audio."}

    duration = len(x) / sr
    if duration < MIN_AUDIO_SEC:
        return {
            "status": "insufficient_audio",
            "speech_sec": round(duration, 1),
            "required_sec": MIN_AUDIO_SEC,
            "message": (f"Only {duration:.1f}s of audio. At least "
                        f"{MIN_AUDIO_SEC:.0f}s is needed — below this the "
                        f"method is unreliable (AUC drops to ~0.57).")
        }

    feats = extract_features(x, sr)
    if feats is None:
        return {"status": "error", "message": "Could not analyse the audio."}

    b = load_model()
    row = np.array([[feats.get(f, 0.0) for f in b["features"]]])
    prob = float(b["model"].predict_proba(row)[0, 1])
    thr = b["threshold"]

    if prob < thr - 0.10:
        band = "typical"
    elif prob < thr:
        band = "borderline"
    else:
        band = "above_typical"

    return {
        "status": "ok",
        "probability": round(prob, 3),
        "threshold": thr,
        "flagged": prob >= thr,
        "band": band,
        "speech_sec": round(duration, 1),
        "features": feats,
        "performance": b["metadata"],
    }
