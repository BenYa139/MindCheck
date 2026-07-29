import streamlit as st
import speech_recognition as sr
import tempfile
import os
import datetime
import numpy as np

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

st.set_page_config(page_title="MindCheck", page_icon="🧠", layout="centered")

# ─────────────────────────────────────────────
# CONTENT
# Cognitive test adapted from MoCA (Nasreddine et al., 2005)
# Speech pause analysis based on Cohen et al. (2026) & Lin et al. (2025)
# ─────────────────────────────────────────────
WORDS = ["face", "velvet", "church", "daisy", "red"]
SENTENCES = [
    "I only know that John is the one to help today",
    "The cat always hid under the couch when dogs were in the room",
]
KEYWORDS = {
    "vehicle":   ["vehicle", "transport", "transportation", "wheels", "ride", "travel", "move"],
    "furniture": ["furniture", "wood", "sit", "house", "home"],
    "watch":     ["watch"],
    "pen":       ["pen", "pencil"],
    "dog":       ["dog"],
}
MAX_SCORE = 30

# Keys that capture speech for pause analysis
SPEECH_KEYS = [
    "fwd", "bwd", "lang1", "lang2",
    "abs1_widget", "abs2_widget",
    "ori_day_name", "ori_date", "ori_month", "ori_year", "ori_season",
    "ori_city",
    "fluency_animals",
    "calc_serial7",
    "naming_watch", "naming_pen", "naming_dog",
    "recall_widget",
]

def current_context():
    now = datetime.datetime.now()
    month = now.month
    season = ("winter" if month in (12,1,2) else
              "spring" if month in (3,4,5) else
              "summer" if month in (6,7,8) else "fall")
    return {
        "day": now.day,
        "weekday": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
        "season": season,
    }

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name
    try:
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            data = r.record(source)
        return r.recognize_google(data, language="en-US")
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.error(f"Speech service error: {e}")
        return None
    finally:
        os.unlink(path)

def similarity_score(spoken, reference):
    spoken_clean = spoken.lower().replace(" ", "")
    matches = sum(1 for w in reference.split() if w.lower() in spoken_clean)
    return matches / max(len(reference.split()), 1)

def digits_in_order(text, sequence):
    return [ch for ch in text if ch.isdigit()] == sequence

def contains_any(text, keywords):
    import re
    text_l = text.lower()
    for k in keywords:
        if re.search(r"\b" + re.escape(k.lower()) + r"\b", text_l):
            return True
    return False

# ─────────────────────────────────────────────
# SPEECH TIMING ANALYSIS
#
# Method validated against 100 DementiaBank Pitt Corpus recordings
# (50 control / 50 dementia, one recording per participant).
#
# Validation results:
#   pause count      control 39.1  vs dementia 51.7   p = 0.021
#   total pause time control 20.0s vs dementia 27.0s  p = 0.016
#   silence duration control 23.5s vs dementia 31.0s  p = 0.014
#   avg pause length control 0.51s vs dementia 0.52s  p = 0.822 (no difference)
#
# Key finding: participants with dementia paused MORE OFTEN, not for longer.
#
# NOTE: an earlier version of this app used a fixed threshold at 8% of peak
# energy and a pacing multiplier. That version was found to saturate: 98 of
# 100 recordings returned an identical score, making it non-functional. It
# has been replaced by the adaptive method below.
# ─────────────────────────────────────────────

FRAME_MS = 30
MIN_PAUSE_SEC = 0.15    # below this = normal gap between words
MAX_PAUSE_SEC = 2.5     # above this = dead air / not the speaker, excluded
NOISE_PERCENTILE = 10   # noise floor estimated from quietest frames
NOISE_MULTIPLIER = 2.5  # speech must exceed noise floor by this factor


def _find_runs(mask, value):
    """Return (start, length) for each contiguous run equal to `value`."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def analyze_speech_timing(audio_bytes):
    """
    Analyse one recording and return pause statistics.

    Returns dict with:
        silence_ratio    fraction of analysed time that was silent
        pause_count      number of hesitation pauses (0.15-2.5 s)
        pause_rate       pauses per minute of analysed speech
        avg_pause        mean pause length in seconds
        analysed_sec     duration analysed after removing dead air
    Returns None if the clip is too short or contains no detectable speech.
    """
    import wave
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            path = f.name

        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if not raw or framerate == 0:
            return None
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            return None

        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if sampwidth == 1:
            samples -= 128.0
        if n_channels > 1:
            usable = (samples.size // n_channels) * n_channels
            samples = samples[:usable].reshape(-1, n_channels).mean(axis=1)

        frame_len = max(int(framerate * FRAME_MS / 1000), 1)
        n_frames = samples.size // frame_len
        if n_frames < 20:          # under ~0.6 s — too short to analyse
            return None

        frames = samples[:n_frames * frame_len].reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))

        # --- adaptive threshold from this clip's own noise floor ---
        noise_floor = np.percentile(rms, NOISE_PERCENTILE)
        if noise_floor <= 0:
            positive = rms[rms > 0]
            noise_floor = positive.min() if positive.size else 1e-9
        threshold = noise_floor * NOISE_MULTIPLIER

        # fallback if the noise floor estimate is unusable
        if (rms >= threshold).sum() < n_frames * 0.02:
            threshold = float(np.max(rms)) * 0.08

        voiced = rms >= threshold
        if voiced.sum() == 0:
            return None

        frame_sec = FRAME_MS / 1000
        min_pause_frames = max(int(MIN_PAUSE_SEC / frame_sec), 1)
        max_pause_frames = int(MAX_PAUSE_SEC / frame_sec)

        # --- classify each silent run ---
        pauses, dead_frames = [], 0
        for start, length in _find_runs(voiced, False):
            at_edge = (start == 0) or (start + length == n_frames)
            if at_edge or length > max_pause_frames:
                dead_frames += length                    # excluded
            elif length >= min_pause_frames:
                pauses.append(length * frame_sec)        # counted

        analysed_frames = n_frames - dead_frames
        if analysed_frames <= 0:
            return None

        analysed_sec = analysed_frames * frame_sec
        silence_ratio = 1.0 - (int(voiced.sum()) / analysed_frames)
        silence_ratio = max(0.0, min(1.0, silence_ratio))

        return {
            "silence_ratio": round(silence_ratio, 4),
            "pause_count": len(pauses),
            "pause_rate": round(len(pauses) / (analysed_sec / 60), 2) if analysed_sec > 0 else 0.0,
            "avg_pause": round(float(np.mean(pauses)), 3) if pauses else 0.0,
            "analysed_sec": round(analysed_sec, 2),
        }

    except Exception:
        return None
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def get_speech_summary():
    """
    Combine timing results across every answer in this session.
    Pooling matters: a single 5-second answer gives a very noisy estimate,
    but 15+ answers together give a usable one.
    """
    ratios, counts, total_sec = [], 0, 0.0
    n_clips = 0
    for key in SPEECH_KEYS:
        res = st.session_state.get(f"{key}_timing")
        if res:
            ratios.append(res["silence_ratio"])
            counts += res["pause_count"]
            total_sec += res["analysed_sec"]
            n_clips += 1
    if n_clips == 0 or total_sec <= 0:
        return None
    return {
        "mean_silence_ratio": sum(ratios) / len(ratios),
        "total_pauses": counts,
        "pause_rate": counts / (total_sec / 60),
        "total_speech_sec": total_sec,
        "n_clips": n_clips,
    }


def pause_level(summary):
    """
    Classify against the DementiaBank control group distribution.

    Thresholds are the 50th and 75th percentile of silence ratio in the
    50 control participants (median 0.381, p75 0.504). The dementia group
    median was 0.466.

    LIMITATION: those percentiles come from ~65-second picture-description
    recordings. This app collects much shorter answers, so values are
    noisier and comparisons are approximate.
    """
    if summary is None:
        return None
    r = summary["mean_silence_ratio"]
    if r < 0.381:
        return "typical"
    elif r < 0.504:
        return "somewhat_elevated"
    else:
        return "elevated"

def get_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")

def ai_grade(question_text, spoken_answer):
    if not ANTHROPIC_SDK_AVAILABLE:
        return None
    api_key = get_api_key()
    if not api_key or not spoken_answer.strip():
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            messages=[{"role": "user", "content":
                f'Grade this spoken answer. Reply YES or NO only.\nQuestion: "{question_text}"\nAnswer: "{spoken_answer}"'}],
        )
        return response.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return None

def voice_input(key):
    """Record audio, transcribe, and store raw pause ratio separately."""
    audio = st.audio_input("🎙️ Record your answer", key=key)
    if audio is not None:
        audio_bytes = audio.read()
        if audio_bytes != st.session_state.get(f"{key}_bytes"):
            st.session_state[f"{key}_bytes"] = audio_bytes
            with st.spinner("Transcribing…"):
                result = transcribe_audio(audio_bytes)
            st.session_state[f"{key}_text"] = result or ""
            if result is None:
                st.error("Could not recognise — please try again.")
            # Speech timing is recorded separately and never alters the MoCA score
            timing = analyze_speech_timing(audio_bytes)
            st.session_state[f"{key}_timing"] = timing
    return st.session_state.get(f"{key}_text", "")

def show_answer(text):
    st.success(f"You said: **{text}**")

# ─────────────────────────────────────────────
# STEP FUNCTIONS (MoCA-aligned)
# ─────────────────────────────────────────────

def step_word_memory():
    st.subheader("📋 Word Memory")
    st.caption("MoCA Domain: Memory — Nasreddine et al., 2005")
    st.write("Memorize these words — you'll be asked again later:")
    cols = st.columns(len(WORDS))
    for i, w in enumerate(WORDS):
        cols[i].markdown(
            f"<div style='text-align:center;font-size:1.4rem;font-weight:700;"
            f"padding:1rem;background:#f0f4ff;border-radius:12px;'>{w}</div>",
            unsafe_allow_html=True
        )
    st.info("Read these carefully, then press **Next**.")

def step_forward_digits():
    st.subheader("🔢 Forward Digit Span")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Say these numbers in the same order:")
    st.markdown(
        "<div style='font-size:2rem;font-weight:700;letter-spacing:0.5rem;"
        "text-align:center;padding:1.5rem;background:#f0f4ff;border-radius:12px;'>"
        "2 – 1 – 8 – 5 – 4</div>", unsafe_allow_html=True
    )
    ans = voice_input("fwd")
    if ans:
        show_answer(ans)
        ok = digits_in_order(ans, ["2","1","8","5","4"])
        st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
        st.session_state["fwd_ok"] = ok

def step_backward_digits():
    st.subheader("🔢 Backward Digit Span")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Say 7 – 4 – 2 in **reverse** order:")
    st.markdown(
        "<div style='font-size:2rem;font-weight:700;letter-spacing:0.5rem;"
        "text-align:center;padding:1.5rem;background:#f0f4ff;border-radius:12px;'>"
        "7 – 4 – 2</div>", unsafe_allow_html=True
    )
    ans = voice_input("bwd")
    if ans:
        show_answer(ans)
        ok = digits_in_order(ans, ["2","4","7"])
        st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
        st.session_state["bwd_ok"] = ok

def make_sentence_step(i, sentence):
    def step():
        st.subheader(f"🗣️ Sentence Repetition ({i}/2)")
        st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
        st.write("Repeat this sentence exactly:")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;font-style:italic;'>\"{sentence}\"</div>",
            unsafe_allow_html=True
        )
        ans = voice_input(f"lang{i}")
        if ans:
            show_answer(ans)
            sc = similarity_score(ans, sentence)
            st.progress(sc, text=f"Match: {sc:.0%}")
            st.session_state[f"lang{i}_score"] = sc
    return step

def make_abstraction_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🧩 Abstraction ({n}/2)")
        st.caption("MoCA Domain: Abstraction — Nasreddine et al., 2005")
        st.write("How are these two things similar?")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(f"abs{n}_widget")
        if ans:
            show_answer(ans)
            ok = ai_grade(question, ans)
            if ok is None:
                ok = contains_any(ans, KEYWORDS[kw_key])
            st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
            st.session_state[f"abs{n}_correct"] = ok
    return step

def make_ori_time_step(question, key, check_fn, truth):
    def step():
        st.subheader("🕐 Orientation to Time")
        st.caption("MoCA Domain: Orientation — Nasreddine et al., 2005")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = check_fn(ans)
            st.write("✅ Correct! (1/1 pt)" if ok else f"❌ It's actually: **{truth}** (0/1 pt)")
            st.session_state[f"{key}_ok"] = ok
    return step

def step_ori_place():
    def step():
        st.subheader("📍 Orientation to Place")
        st.caption("MoCA Domain: Orientation — Nasreddine et al., 2005")
        st.caption("Your answer is recorded for a reviewer to verify.")
        st.markdown(
            "<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            "border-radius:12px;'>What city or town are you in right now?</div>",
            unsafe_allow_html=True
        )
        ans = voice_input("ori_city")
        if ans:
            show_answer(ans)
            st.session_state["ori_city_answered"] = True
    return step

def step_fluency_animals():
    st.subheader("🦁 Verbal Fluency")
    st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
    st.write("⏱️ Name as many **animals** as you can in 1 minute:")
    ans = voice_input("fluency_animals")
    if ans:
        show_answer(ans)
        count = len(set(ans.lower().split()))
        st.write(f"📊 Approximate word count: **{count}** (need ≥11 for full point)")
        st.session_state["fluency_animals_count"] = count

def step_calculation():
    st.subheader("🧮 Calculation")
    st.caption("MoCA Domain: Attention — Nasreddine et al., 2005")
    st.write("Starting at 100, keep subtracting 7 and say **5 results** in a row:")
    st.markdown(
        "<div style='font-size:1.4rem;text-align:center;padding:1rem;"
        "background:#f0f4ff;border-radius:12px;'>100 → 93 → 86 → 79 → 72 → 65</div>",
        unsafe_allow_html=True
    )
    ans = voice_input("calc_serial7")
    if ans:
        show_answer(ans)
        expected = ["93","86","79","72","65"]
        spoken = [t for t in ans.replace(",", " ").split() if t.isdigit()]
        correct = sum(1 for e in expected if e in spoken)
        pts = 3 if correct >= 4 else 2 if correct == 3 else 1 if correct in (1,2) else 0
        st.write(f"📊 {correct}/5 correct → **{pts}/3 pts**")
        st.session_state["calc_correct_count"] = correct
        st.session_state["calc_pts"] = pts

def make_naming_step(n, question, key, kw_key):
    def step():
        st.subheader(f"🏷️ Naming ({n}/3)")
        st.caption("MoCA Domain: Language — Nasreddine et al., 2005")
        st.markdown(
            f"<div style='font-size:1.2rem;padding:1.5rem;background:#f0f4ff;"
            f"border-radius:12px;'>{question}</div>", unsafe_allow_html=True
        )
        ans = voice_input(key)
        if ans:
            show_answer(ans)
            ok = ai_grade(question, ans)
            if ok is None:
                ok = contains_any(ans, KEYWORDS[kw_key])
            st.write("✅ Correct! (1/1 pt)" if ok else "❌ Not quite. (0/1 pt)")
            st.session_state[f"{key}_ok"] = ok
    return step

def step_delayed_recall():
    st.subheader("🧠 Delayed Recall")
    st.caption("MoCA Domain: Memory — Nasreddine et al., 2005")
    st.write("Say as many words as you can remember from the very beginning:")
    ans = voice_input("recall_widget")
    if ans:
        show_answer(ans)
        found = [w for w in WORDS if w.lower() in ans.lower()]
        pct = len(found) / len(WORDS)
        st.progress(pct, text=f"Accuracy: {pct:.0%}")
        st.write(f"📊 {len(found)} / {len(WORDS)} words ({len(found)}/5 pts)")
        st.session_state["recall_score_count"] = len(found)

def step_results():
    st.subheader("📊 Results")

    # ── COMPUTE MOCA SCORE ──────────────────────
    score = 0
    details = []

    fwd_pt = 1 if st.session_state.get("fwd_ok") else 0
    score += fwd_pt
    details.append(f"{'✅' if fwd_pt else '❌'} Forward Digits: {fwd_pt}/1 pt")

    bwd_pt = 1 if st.session_state.get("bwd_ok") else 0
    score += bwd_pt
    details.append(f"{'✅' if bwd_pt else '❌'} Backward Digits: {bwd_pt}/1 pt")

    for i in range(1, 3):
        s = st.session_state.get(f"lang{i}_score", 0.0)
        pt = 1 if s >= 0.6 else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Sentence {i}: {pt}/1 pt ({s:.0%} match)")

    for i in range(1, 3):
        pt = 1 if st.session_state.get(f"abs{i}_correct") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Abstraction {i}: {pt}/1 pt")

    for key in ["ori_day_name","ori_date","ori_month","ori_year","ori_season"]:
        pt = 1 if st.session_state.get(f"{key}_ok") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Time ({key}): {pt}/1 pt")

    pt = 1 if st.session_state.get("ori_city_answered") else 0
    score += pt
    details.append(f"{'✅' if pt else '❌'} Place (city): {pt}/1 pt")

    count = st.session_state.get("fluency_animals_count", 0)
    pt = 1 if count >= 11 else 0
    score += pt
    details.append(f"{'✅' if pt else '❌'} Verbal Fluency: {count} animals → {pt}/1 pt")

    calc_pts = st.session_state.get("calc_pts", 0)
    score += calc_pts
    details.append(f"🧮 Calculation: {calc_pts}/3 pts")

    for key in ["naming_watch","naming_pen","naming_dog"]:
        pt = 1 if st.session_state.get(f"{key}_ok") else 0
        score += pt
        details.append(f"{'✅' if pt else '❌'} Naming ({key}): {pt}/1 pt")

    found = st.session_state.get("recall_score_count", 0)
    score += found
    details.append(f"🧠 Delayed Recall: {found}/5 pts")

    # ── SPEECH TIMING SUMMARY ────────────────────
    speech = get_speech_summary()
    s_level = pause_level(speech)

    # ── COGNITIVE BAND (MoCA thresholds) ─────────
    if score >= 26:
        cog_label = "🟢 Typical range"
        cog_desc = "Score is in the typical range (26-30)"
    elif score >= 18:
        cog_label = "🟡 Below typical"
        cog_desc = "Score is below the typical range (18-25)"
    else:
        cog_label = "🔴 Well below typical"
        cog_desc = "Score is well below the typical range (under 18)"

    # ── SPEECH BAND ──────────────────────────────
    if speech is None:
        speech_val, speech_label, speech_desc = "—", "Not available", "No usable recordings"
    else:
        speech_val = f"{speech['mean_silence_ratio']:.0%}"
        if s_level == "typical":
            speech_label = "🟢 Typical range"
            speech_desc = "Pausing was within the typical range"
        elif s_level == "somewhat_elevated":
            speech_label = "🟡 Somewhat above typical"
            speech_desc = "Slightly more pausing than typical"
        else:
            speech_label = "🟠 Above typical"
            speech_desc = "More pausing than typical for this task"

    # ── DISPLAY ──────────────────────────────────
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style='text-align:center;padding:1.5rem;background:#f0f4ff;
            border-radius:16px;margin-bottom:1rem;'>
                <div style='font-size:0.85rem;color:#666;margin-bottom:0.5rem;'>
                    🧠 Cognitive Score (MoCA-based)
                </div>
                <div style='font-size:2.5rem;font-weight:700;'>{score}/{MAX_SCORE}</div>
                <div style='font-size:1rem;margin-top:0.5rem;'>{cog_label}</div>
                <div style='font-size:0.8rem;color:#555;margin-top:0.25rem;'>{cog_desc}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='text-align:center;padding:1.5rem;background:#f0f4ff;
            border-radius:16px;margin-bottom:1rem;'>
                <div style='font-size:0.85rem;color:#666;margin-bottom:0.5rem;'>
                    🎙️ Speech Timing
                </div>
                <div style='font-size:2.5rem;font-weight:700;'>{speech_val}</div>
                <div style='font-size:1rem;margin-top:0.5rem;'>{speech_label}</div>
                <div style='font-size:0.8rem;color:#555;margin-top:0.25rem;'>{speech_desc}</div>
            </div>
        """, unsafe_allow_html=True)

    # ── OVERALL RESULT ───────────────────────────
    cog_concern = score < 26
    speech_concern = s_level in ("somewhat_elevated", "elevated")

    if not cog_concern and not speech_concern:
        r_bg, r_border = "#E8F5E9", "#4CAF50"
        r_icon, r_title = "🟢", "Low Risk"
        r_line = "Nothing in today's check stood out."
        r_action = "Keep an eye on things, and check again in a few months if you like."
    elif cog_concern and speech_concern:
        r_bg, r_border = "#FDECEA", "#E53935"
        r_icon, r_title = "🔴", "High Risk"
        r_line = "Both parts of today's check were outside the usual range."
        r_action = "Please make an appointment to talk with a doctor."
    else:
        r_bg, r_border = "#FFF8E1", "#FFA726"
        r_icon, r_title = "🟡", "Moderate Risk"
        r_line = "One part of today's check was outside the usual range."
        r_action = "It would be worth mentioning this to a doctor at your next visit."

    st.markdown(
f"""<div style='text-align:center;padding:2rem 1.5rem;background:{r_bg};border:3px solid {r_border};border-radius:20px;margin:1.5rem 0;'>
<div style='font-size:1.1rem;color:#555;margin-bottom:0.5rem;'>Today's Result</div>
<div style='font-size:2.6rem;font-weight:800;line-height:1.2;'>{r_icon} {r_title}</div>
<div style='font-size:1.25rem;color:#333;margin-top:1rem;'>{r_line}</div>
<div style='font-size:1.25rem;color:#333;margin-top:0.5rem;font-weight:600;'>{r_action}</div>
</div>""", unsafe_allow_html=True)

    # ── RISK GAUGE ───────────────────────────────
    # The needle points to the middle of the band, because this result is a
    # 3-level category, not a continuous 0-100 score. Showing a precise
    # number would imply accuracy this method does not have.
    if r_title == "Low Risk":
        needle_x, needle_y, gauge_col = 113, 130, "#4CAF50"
    elif r_title == "Moderate Risk":
        needle_x, needle_y, gauge_col = 200, 80, "#FFA726"
    else:
        needle_x, needle_y, gauge_col = 287, 130, "#E53935"

    st.markdown(
f"""<div style='text-align:center;margin:1.5rem 0;'>
<svg viewBox="0 0 400 250" style="width:100%;max-width:420px;height:auto;">
<path d="M 70 180 A 130 130 0 0 1 135 67" stroke="#4CAF50" stroke-width="34" fill="none" opacity="{1.0 if r_title=='Low Risk' else 0.25}"/>
<path d="M 135 67 A 130 130 0 0 1 265 67" stroke="#FFA726" stroke-width="34" fill="none" opacity="{1.0 if r_title=='Moderate Risk' else 0.25}"/>
<path d="M 265 67 A 130 130 0 0 1 330 180" stroke="#E53935" stroke-width="34" fill="none" opacity="{1.0 if r_title=='High Risk' else 0.25}"/>
<text x="72" y="214" font-size="20" font-weight="700" fill="#4CAF50" text-anchor="middle">LOW</text>
<text x="200" y="34" font-size="20" font-weight="700" fill="#FFA726" text-anchor="middle">MODERATE</text>
<text x="330" y="214" font-size="20" font-weight="700" fill="#E53935" text-anchor="middle">HIGH</text>
<line x1="200" y1="180" x2="{needle_x}" y2="{needle_y}" stroke="#37474F" stroke-width="7" stroke-linecap="round"/>
<circle cx="200" cy="180" r="15" fill="#37474F"/>
<text x="200" y="243" font-size="27" font-weight="800" fill="{gauge_col}" text-anchor="middle">{r_title.upper()}</text>
</svg>
</div>""", unsafe_allow_html=True)

    # ── PLAIN-LANGUAGE EXPLANATION ───────────────
    cog_plain = ("Your answers were mostly correct." if score >= 26
                 else "You got some of the questions wrong." if score >= 18
                 else "Many of the questions were answered incorrectly.")
    speech_plain = ("You spoke smoothly, without many long pauses."
                    if s_level == "typical" else
                    "You paused a little more than most people do."
                    if s_level == "somewhat_elevated" else
                    "You paused quite a lot while speaking.")

    st.markdown(
f"""<div style='padding:1.5rem;background:#FFF8E6;border-left:5px solid #E8A33D;border-radius:10px;margin:1rem 0;font-size:1.1rem;line-height:1.8;color:#333;'>
<div style='font-weight:700;font-size:1.25rem;margin-bottom:0.75rem;'>What these two numbers mean</div>
<b>🧠 Memory and thinking score &mdash; {score} out of 30</b><br>
This counts how many questions you answered correctly. {cog_plain} A score of 26 or more is the usual range.
<br><br>
<b>🎙️ Speech timing &mdash; {speech_val}</b><br>
This measures how much of the time you were quiet while speaking. {speech_plain} Everyone pauses when they talk, and that is normal.
<br><br>
<div style='background:#ffffff;padding:1rem;border-radius:8px;'>
<b>Please remember:</b> this is a simple check made by students, not a medical test. It is right about <b>6 times out of 10</b>, so it can easily be wrong about you. Being tired, nervous, or in a noisy room can change your result.
<br><br>
<b>Only a doctor can tell you about your memory or your health.</b>
</div>
</div>""", unsafe_allow_html=True)

    with st.expander("📋 Cognitive Score Breakdown"):
        for d in details:
            st.write(d)

    if speech is not None:
        with st.expander("📋 Speech Timing Breakdown"):
            st.write(f"**Average silence ratio:** {speech['mean_silence_ratio']:.1%}")
            st.write(f"**Total pauses detected:** {speech['total_pauses']}")
            st.write(f"**Pause rate:** {speech['pause_rate']:.1f} per minute")
            st.write(f"**Speech analysed:** {speech['total_speech_sec']:.1f} s "
                     f"across {speech['n_clips']} recordings")
            st.markdown("---")
            st.write("**Reference values** (DementiaBank Pitt Corpus, n=100):")
            st.write("- Control group median silence ratio: 38.1%")
            st.write("- Dementia group median silence ratio: 46.6%")
            st.write("- Bands used here: typical <38.1% | somewhat elevated 38.1-50.4% | elevated >50.4%")
            st.caption(
                "Note: reference values come from ~65-second picture-description "
                "recordings. This app records much shorter answers, so values are "
                "noisier and the comparison is approximate."
            )

    st.markdown("---")
    st.caption("⚠️ This is a student research prototype. It is NOT a medical device and cannot diagnose Alzheimer's disease or any other condition. Please consult a qualified clinician for any health concern.")
    st.caption("📚 Cognitive items adapted from: Nasreddine ZS et al. (2005). *Journal of the American Geriatrics Society*, 53(4), 695-699.")
    st.caption("📚 Speech timing method validated on the DementiaBank Pitt Corpus (n=100, 50 control / 50 dementia). Background: Cohen et al. (2026), *Journal of the International Neuropsychological Society*, 32(1), 24-31; Lin et al. (2025), *Alzheimer's & Dementia*, doi:10.1002/alz.086309")

    if st.button("🔁 Start Over", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["step"] = 0
        st.rerun()

@st.dialog("⚠️ Important Disclaimer")
def show_disclaimer():
    """Modal shown when the sidebar Disclaimer button is pressed."""
    st.markdown("""
**MindCheck is a student research prototype. It is not a medical device.**

This tool cannot diagnose Alzheimer's disease, dementia, or any other
medical condition. It has not been reviewed or approved by any medical
authority.

**How accurate is it?**

The speech model was tested on 100 recordings from the DementiaBank Pitt
Corpus (50 people with dementia, 50 without). It gave the correct answer
about **61% of the time** — better than guessing, but wrong roughly 4 times
in every 10.

**What can change your result?**

Being tired, feeling nervous, background noise, microphone quality, or
simply an unfamiliar task can all change your score. A result outside the
usual range does not mean something is wrong.

**Language**

The system works in English only. It has not been tested on Thai speakers.

**Your privacy**

No personal information is collected. Your recordings are processed only
during the session and are not saved anywhere. Closing the page erases
everything.

**If you are worried about your memory or thinking, please talk to a doctor.**
This tool cannot answer that question and was not designed to.
    """)
    st.caption("Cognitive items adapted from Nasreddine ZS et al. (2005), *Journal of the American Geriatrics Society*, 53(4), 695-699.")
    if st.button("Close", use_container_width=True):
        st.rerun()


def step_feedback():
    st.subheader("⭐ Rate MindCheck")
    st.write("How useful was this check for you?")

    rating = st.feedback("stars", key="user_rating")

    if rating is not None:
        stars = rating + 1     # st.feedback returns 0-4
        messages = {
            1: "Thank you — we're sorry it wasn't useful. Your feedback helps us improve.",
            2: "Thank you for the honest rating. We know there is a lot to improve.",
            3: "Thank you — we appreciate the balanced feedback.",
            4: "Thank you! We're glad it was helpful.",
            5: "Thank you so much! We're glad it was helpful.",
        }
        st.success(f"{'⭐' * stars}  {messages[stars]}")

    st.write("")
    st.text_area(
        "Any comments? (optional)",
        placeholder="What worked well? What was confusing?",
        key="user_comment",
        height=110,
    )
    st.caption(
        "Note: this is a prototype. Ratings and comments are not saved or "
        "transmitted anywhere — they are cleared when you close the page."
    )

    st.markdown("---")
    if st.button("🔁 Start a New Check", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["step"] = 0
        st.rerun()


# ─────────────────────────────────────────────
# BUILD STEPS LIST
# ─────────────────────────────────────────────
ctx = current_context()

STEPS = [
    step_word_memory,
    step_forward_digits,
    step_backward_digits,
    make_sentence_step(1, SENTENCES[0]),
    make_sentence_step(2, SENTENCES[1]),
    make_abstraction_step(1, "How are a Train and a Bicycle similar?", "abs1", "vehicle"),
    make_abstraction_step(2, "How are a Table and a Chair similar?", "abs2", "furniture"),
    make_ori_time_step("What day of the week is it today?", "ori_day_name",
                       lambda a: ctx["weekday"].lower() in a.lower(), ctx["weekday"]),
    make_ori_time_step("What is today's date?", "ori_date",
                       lambda a: str(ctx["day"]) in a, ctx["day"]),
    make_ori_time_step("What month is it?", "ori_month",
                       lambda a: ctx["month"].lower() in a.lower(), ctx["month"]),
    make_ori_time_step("What year is it?", "ori_year",
                       lambda a: str(ctx["year"]) in a, ctx["year"]),
    make_ori_time_step("What season is it?", "ori_season",
                       lambda a: ctx["season"].lower() in a.lower(), ctx["season"]),
    step_ori_place(),
    step_fluency_animals,
    step_calculation,
    make_naming_step(1, "What do you call the object worn on the wrist that tells time?", "naming_watch", "watch"),
    make_naming_step(2, "What do you call the object used for writing?", "naming_pen", "pen"),
    make_naming_step(3, "What do you call the pet that barks and guards the house?", "naming_dog", "dog"),
    step_delayed_recall,
    step_results,
    step_feedback,
]

TOTAL_STEPS = len(STEPS)

if "step" not in st.session_state:
    st.session_state["step"] = 0

step_idx = st.session_state["step"]

st.markdown("""
    <style>
    div[data-testid="stAudioInput"] { transform: scale(1.3); transform-origin: left top; margin-bottom: 24px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**🧠 MindCheck**")
    st.caption("Cognitive + Speech Analysis")
    st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

    st.markdown("---")

    # Language selector. Only English is available: the model was trained
    # entirely on English speech, so offering other languages would imply
    # a capability the system does not have.
    st.selectbox(
        "🌐 Language",
        ["English"],
        index=0,
        help="Only English is available. The speech model was trained on English recordings and has not been validated for other languages.",
    )

    if st.button("⚠️ Disclaimer", use_container_width=True):
        show_disclaimer()

st.progress(step_idx / (TOTAL_STEPS - 1), text=f"Step {step_idx + 1} / {TOTAL_STEPS}")

STEPS[step_idx]()

if step_idx < TOTAL_STEPS - 1:
    st.write("")
    col1, col2 = st.columns([1, 2])
    with col1:
        if step_idx > 0:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state["step"] -= 1
                st.rerun()
    with col2:
        if st.button("➡️ Next", use_container_width=True, type="primary"):
            st.session_state["step"] += 1
            st.rerun()
