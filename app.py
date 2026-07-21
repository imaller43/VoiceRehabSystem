# app.py - Voice Rehabilitation AI Backend (Improved DSP Engine)
import os
import tempfile
import warnings
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from scipy.signal import medfilt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ==============================================================
# SAMBUNGAN MONGODB PANGKALAN DATA PESAKIT
# ==============================================================
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

MONGO_URI = "mongodb://localhost:27017/" 
try:
    client = MongoClient(MONGO_URI)
    db = client['voice_rehab_db']
    users_collection = db['patients']
    print("✅ Pangkalan Data MongoDB Berjaya Disambungkan!")
except Exception as e:
    print(f"❌ Ralat MongoDB: {e}")

# ==============================================================
# API PENGESAHAN (AUTHENTICATION) & KATA LALUAN
# ==============================================================
@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Sila berikan username dan kata laluan"}), 400
    if users_collection.find_one({"username": username}):
        return jsonify({"error": "Nama pengguna telah wujud!"}), 409

    hashed_password = generate_password_hash(password)
    new_user = {
        "username": username,
        "password": hashed_password,
        "diagnostic_status": {"level": "", "abnormalStage": 1, "tested": False},
        "rehab_progress": {"stage1_streak": 0, "stage2_streak": 0, "stage3_streak": 0},
        "training_history": []
    }
    users_collection.insert_one(new_user)
    return jsonify({"message": "Pendaftaran berjaya"}), 201

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = users_collection.find_one({"username": username})
    if user and check_password_hash(user['password'], password):
        return jsonify({"message": "Log masuk berjaya", "username": user['username']}), 200
    return jsonify({"error": "Nama pengguna atau kata laluan salah!"}), 401

@app.route('/api/reset_password', methods=['POST'])
def reset_password():
    data = request.json
    username = data.get('username')
    new_password = data.get('new_password')

    user = users_collection.find_one({"username": username})
    if user:
        hashed_password = generate_password_hash(new_password)
        users_collection.update_one({"username": username}, {"$set": {"password": hashed_password}})
        return jsonify({"message": "Kata laluan berjaya ditukar!"}), 200
    return jsonify({"error": "Pengguna tidak wujud."}), 404

# ==============================================================
# BAHAGIAN 1: AI DIAGNOSTIK (SVM KLASIFIKASI KESIHATAN SUARA)
# ==============================================================
MODEL_PATH = 'model/model_diagnostik_suara4.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model AI Diagnostik Berjaya Dimuatkan! (Jenis: {type(model).__name__})")
except Exception as e:
    print(f"❌ Ralat memuatkan model: {e}")
    model = None

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path)
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(audio_trimmed) > (sample_rate * 0.5):
            audio = audio_trimmed
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Ralat pengekstrakan: {e}")
        return None

@app.route('/api/diagnose', methods=['POST'])
def diagnose_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Tiada fail audio dijumpai"}), 400

    audio_file = request.files['audio']
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_diagnostic.wav")
    audio_file.save(temp_path)

    features = extract_features(temp_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if features is None or model is None:
        return jsonify({"status": "error", "level": "NORMAL", "message": "Gagal memproses audio"}), 500

    print("\n" + "="*30)
    print("🧠 LAPORAN AI DIAGNOSTIK 🧠")
    print("="*30)
    
    try:
        # 1. Dapatkan peratusan dari otak statistik
        probabilities = model.predict_proba([features])[0]
        classes = model.classes_
        
        # 2. Cetak peratusan di terminal
        for cls, prob in zip(classes, probabilities):
            print(f"Tahap {cls}: {prob*100:.2f}%")
            
        # 3. KEPUTUSAN AKHIR: Ambil kelas yang ada peratusan paling tinggi
        max_index = np.argmax(probabilities) 
        detected_level = str(classes[max_index]).upper()
        
    except AttributeError:
        # Jika model tiada sokongan peratusan (fallback)
        prediction = model.predict([features])
        detected_level = str(prediction[0]).upper()

    # Bersihkan nama jika ada nombor atau simbol (cth: "1.ABNORMAL" -> "ABNORMAL")
    if "ABNORMAL" in detected_level: 
        detected_level = "ABNORMAL"
    elif "NORMAL" in detected_level: 
        detected_level = "NORMAL"

    print(f"👉 KEPUTUSAN AKHIR: {detected_level}")
    print("="*30 + "\n")

    return jsonify({"status": "success", "level": detected_level, "message": "Audio dianalisis oleh AI"})


# ==============================================================
# BAHAGIAN 2: DSP HELPER FUNCTIONS
# ==============================================================

def extract_f0_robust(y, sr):
    """
    Extract fundamental frequency (F0) using PYIN algorithm.
    PYIN is more accurate than piptrack for voiced speech/singing.
    Returns: f0 array with NaN for unvoiced frames, times array.
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz  (low bass)
            fmax=librosa.note_to_hz('C6'),   # ~1047 Hz (high soprano)
            sr=sr
        )
        return f0, voiced_flag
    except Exception as e:
        print(f"PYIN error, falling back to piptrack: {e}")
        return extract_f0_piptrack(y, sr)

def extract_f0_piptrack(y, sr):
    """Fallback F0 extraction using piptrack."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=500)
    f0 = []
    voiced = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            f0.append(pitch)
            voiced.append(True)
        else:
            f0.append(np.nan)
            voiced.append(False)
    return np.array(f0), np.array(voiced)

def get_voiced_f0(f0, voiced_flag, min_pitch=60, max_pitch=800):
    """Filter to only keep reliably voiced frames within human pitch range."""
    voiced_pitches = []
    for pitch, is_voiced in zip(f0, voiced_flag):
        if is_voiced and not np.isnan(pitch) and min_pitch < pitch < max_pitch:
            voiced_pitches.append(pitch)
    return np.array(voiced_pitches)

def voiced_ratio(voiced_flag):
    """Returns fraction of frames that are voiced (0.0 to 1.0)."""
    if len(voiced_flag) == 0:
        return 0.0
    return np.sum(voiced_flag) / len(voiced_flag)

def smooth_f0(f0_array, window=5):
    """Apply median filter to remove spurious pitch jumps."""
    if len(f0_array) < window:
        return f0_array
    return medfilt(f0_array, kernel_size=window)

def check_minimum_voice(voiced_pitches, voiced_ratio_val, min_frames=20, min_ratio=0.3):
    """Returns True if there is enough voiced content to evaluate."""
    return len(voiced_pitches) >= min_frames and voiced_ratio_val >= min_ratio


# ==============================================================
# BAHAGIAN 3: EXERCISE SCORING FUNCTIONS
# ==============================================================

def score_gumaman(y, sr):
    """
    PERINGKAT 1 - Gumaman (Humming)
    Goal: Sustain a stable, steady pitch for the full duration.

    Scoring based on:
    1. Voice presence ratio (did they actually hum the whole time?)
    2. Pitch stability (low std deviation = good)
    3. Continuity (no large gaps or breaks in voice)
    """
    f0, voiced_flag = extract_f0_robust(y, sr)
    v_ratio = voiced_ratio(voiced_flag)
    voiced_pitches = get_voiced_f0(f0, voiced_flag)

    print(f"  [Gumaman] Voiced ratio: {v_ratio:.2f}, Voiced frames: {len(voiced_pitches)}")

    # Penalty: not enough voice detected
    if not check_minimum_voice(voiced_pitches, v_ratio, min_frames=15, min_ratio=0.35):
        print(f"  [Gumaman] Suara terlalu pendek/senyap.")
        return 60

    # Smooth to remove mic artifacts
    smoothed = smooth_f0(voiced_pitches)

    # 1. Pitch stability: lower std = better
    pitch_std = np.std(smoothed)
    # Map: std=2Hz → 98pts, std=10Hz → 85pts, std=25Hz → 70pts, std=40Hz → 60pts
    stability_score = np.interp(pitch_std, [2, 10, 25, 40], [98, 85, 70, 60])
    stability_score = float(np.clip(stability_score, 60, 98))

    # 2. Voice presence bonus: reward sustaining voice throughout
    # Ideal: >80% of duration is voiced
    presence_bonus = np.interp(v_ratio, [0.35, 0.6, 0.8, 1.0], [0, 3, 6, 10])

    # 3. Continuity: penalise large pitch jumps (breaks/restarting)
    diffs = np.abs(np.diff(smoothed))
    large_jumps = np.sum(diffs > 30)  # >30Hz jump = break
    continuity_penalty = min(large_jumps * 3, 15)

    raw = stability_score + presence_bonus - continuity_penalty
    final = int(np.clip(raw, 60, 98))

    print(f"  [Gumaman] PitchStd={pitch_std:.1f}Hz, StabilityScore={stability_score:.1f}, PresenceBonus={presence_bonus:.1f}, JumpPenalty={continuity_penalty}")
    print(f"  [Gumaman] 🎯 Markah Akhir: {final}%")
    return final


def score_siren(y, sr):
    """
    PERINGKAT 2 - Luncuran Nada Siren
    Goal: Smoothly glide pitch UP then DOWN (or down-up), covering a wide range.

    Scoring based on:
    1. Pitch range covered (wider = better, up to ~2 octaves)
    2. Smoothness of glide (no sudden jumps, continuous sweep)
    3. Directionality: must go UP then DOWN (or reverse) — not just random fluctuation
    """
    f0, voiced_flag = extract_f0_robust(y, sr)
    v_ratio = voiced_ratio(voiced_flag)
    voiced_pitches = get_voiced_f0(f0, voiced_flag)

    print(f"  [Siren] Voiced ratio: {v_ratio:.2f}, Voiced frames: {len(voiced_pitches)}")

    if not check_minimum_voice(voiced_pitches, v_ratio, min_frames=20, min_ratio=0.35):
        print(f"  [Siren] Suara terlalu pendek/senyap.")
        return 60

    smoothed = smooth_f0(voiced_pitches)

    # 1. Pitch range (ignore extreme 5% to filter noise)
    p5  = np.percentile(smoothed, 5)
    p95 = np.percentile(smoothed, 95)
    pitch_range_hz = p95 - p5
    # Map: 50Hz → 60pts, 120Hz → 75pts, 200Hz → 88pts, 300Hz+ → 97pts
    range_score = np.interp(pitch_range_hz, [50, 120, 200, 300], [60, 75, 88, 97])

    # 2. Smoothness: penalise abrupt jumps between consecutive frames
    diffs = np.abs(np.diff(smoothed))
    # For a 6s siren, ~150 frames; allow average jump of ~3Hz/frame for smooth glide
    mean_jump = np.mean(diffs)
    # Smooth glide: mean_jump ~3Hz, rough/choppy: ~20Hz
    smoothness_score = np.interp(mean_jump, [2, 5, 15, 30], [10, 8, 3, 0])

    # 3. Directionality: check for a clear arc (up-then-down or down-then-up)
    # Split into first half and second half — slopes should be opposite signs
    half = len(smoothed) // 2
    if half > 5:
        first_half  = smoothed[:half]
        second_half = smoothed[half:]
        slope1 = np.polyfit(range(len(first_half)),  first_half,  1)[0]
        slope2 = np.polyfit(range(len(second_half)), second_half, 1)[0]
        # Opposite signs = good arc (siren shape)
        has_arc = (slope1 > 0 and slope2 < 0) or (slope1 < 0 and slope2 > 0)
        directionality_bonus = 8 if has_arc else 0
        print(f"  [Siren] Slope1={slope1:.2f}, Slope2={slope2:.2f}, Arc={has_arc}")
    else:
        directionality_bonus = 0

    raw = range_score + smoothness_score + directionality_bonus
    final = int(np.clip(raw, 60, 98))

    print(f"  [Siren] Range={pitch_range_hz:.1f}Hz, RangeScore={range_score:.1f}, Smoothness={smoothness_score:.1f}, DirBonus={directionality_bonus}")
    print(f"  [Siren] 🎯 Markah Akhir: {final}%")
    return final


def score_messa_di_voce(y, sr):
    """
    PERINGKAT 3 - Messa di Voce
    Goal: Sustain a single pitch, gradually increase volume (crescendo),
          then gradually decrease volume (diminuendo). Shape: soft→LOUD→soft.

    Scoring based on:
    1. Pitch stability (must hold same note throughout)
    2. Dynamic range (difference between loudest and softest)
    3. Shape correctness: volume must RISE then FALL (not flat, not random)
    """
    f0, voiced_flag = extract_f0_robust(y, sr)
    v_ratio = voiced_ratio(voiced_flag)
    voiced_pitches = get_voiced_f0(f0, voiced_flag)

    print(f"  [Messa] Voiced ratio: {v_ratio:.2f}, Voiced frames: {len(voiced_pitches)}")

    if not check_minimum_voice(voiced_pitches, v_ratio, min_frames=20, min_ratio=0.35):
        print(f"  [Messa] Suara terlalu pendek/senyap.")
        return 60

    # 1. Pitch stability (must hold a single note, not wander)
    smoothed_f0 = smooth_f0(voiced_pitches)
    pitch_std = np.std(smoothed_f0)
    stability_score = float(np.interp(pitch_std, [2, 10, 25, 45], [20, 17, 12, 8]))

    # 2. Dynamic range (RMS energy envelope)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_smooth = medfilt(rms, kernel_size=5)
    rms_p5  = np.percentile(rms_smooth, 5)
    rms_p95 = np.percentile(rms_smooth, 95)
    rms_range = rms_p95 - rms_p5

    # Map RMS range to score: 0.02 → 60pts, 0.08 → 75pts, 0.20 → 90pts
    dynamic_score = float(np.interp(rms_range, [0.02, 0.08, 0.20, 0.35], [60, 75, 90, 97]))

    # 3. Shape: crescendo then diminuendo
    # Divide RMS into thirds: beginning, peak zone, end
    n = len(rms_smooth)
    if n >= 12:
        third = n // 3
        start_avg  = np.mean(rms_smooth[:third])
        middle_avg = np.mean(rms_smooth[third:2*third])
        end_avg    = np.mean(rms_smooth[2*third:])

        # Ideal: middle > start and middle > end (mountain shape)
        is_mountain = (middle_avg > start_avg) and (middle_avg > end_avg)
        # Also reward if it monotonically rises then falls
        up_then_down = (middle_avg > start_avg * 1.1) and (middle_avg > end_avg * 1.1)

        if up_then_down:
            shape_bonus = 12
        elif is_mountain:
            shape_bonus = 6
        else:
            shape_bonus = 0

        print(f"  [Messa] RMS Start={start_avg:.4f}, Mid={middle_avg:.4f}, End={end_avg:.4f}, Mountain={is_mountain}")
    else:
        shape_bonus = 0

    # Combine: dynamic range is most important for Messa
    raw = dynamic_score + stability_score + shape_bonus
    final = int(np.clip(raw, 60, 98))

    print(f"  [Messa] PitchStd={pitch_std:.1f}Hz, StabilityScore={stability_score:.1f}, DynamicScore={dynamic_score:.1f}, ShapeBonus={shape_bonus}")
    print(f"  [Messa] 🎯 Markah Akhir: {final}%")
    return final


def score_skala_vokal(y, sr):
    """
    PERINGKAT 3 - Skala Vokal (Vocal Scale)
    Goal: Sing a stepwise scale (do-re-mi-fa-sol or similar).
          Distinct pitch steps, correct sequence, smooth transitions.

    Scoring based on:
    1. Number of distinct pitch steps detected
    2. Whether the steps are approximately equal intervals (semitones)
    3. Voice presence (must sing all notes, no skipped gaps)
    """
    f0, voiced_flag = extract_f0_robust(y, sr)
    v_ratio = voiced_ratio(voiced_flag)
    voiced_pitches = get_voiced_f0(f0, voiced_flag)

    print(f"  [Skala] Voiced ratio: {v_ratio:.2f}, Voiced frames: {len(voiced_pitches)}")

    if not check_minimum_voice(voiced_pitches, v_ratio, min_frames=20, min_ratio=0.30):
        print(f"  [Skala] Suara terlalu pendek/senyap.")
        return 60

    smoothed = smooth_f0(voiced_pitches)

    # Convert to MIDI notes (semitone scale) for step detection
    midi_notes = librosa.hz_to_midi(smoothed)

    # Segment into stable pitch plateaus (each plateau = one scale note)
    # Use a simple diff-based segmentation
    diff_midi = np.abs(np.diff(midi_notes))
    # Frame is a "transition" if pitch changes by >1 semitone
    transitions = np.where(diff_midi > 1.2)[0]

    # Count distinct stable segments
    segment_midis = []
    prev = 0
    for t in transitions:
        seg = midi_notes[prev:t+1]
        if len(seg) >= 3:
            segment_midis.append(np.median(seg))
        prev = t + 1
    # Last segment
    if prev < len(midi_notes) - 2:
        segment_midis.append(np.median(midi_notes[prev:]))

    num_steps = len(segment_midis)
    print(f"  [Skala] Distinct pitch steps detected: {num_steps}")

    # Score: reward getting close to 5–8 distinct scale notes
    # Fewer than 3 = very poor; 5+ = good; 7–8 = excellent
    step_score = np.interp(num_steps, [1, 3, 5, 7, 9], [60, 70, 82, 92, 97])

    # Check if steps are roughly equidistant (equal temperament scale)
    if len(segment_midis) >= 3:
        intervals = np.diff(segment_midis)
        # For a major scale, intervals are mostly 2 or 1 semitone (whole/half steps)
        # Measure consistency: std of intervals should be low for a real scale
        interval_std = np.std(intervals)
        regularity_bonus = float(np.interp(interval_std, [0.5, 1.5, 3.0, 6.0], [10, 7, 3, 0]))
        print(f"  [Skala] Interval std={interval_std:.2f} semitones, Bonus={regularity_bonus:.1f}")
    else:
        regularity_bonus = 0

    # Voice presence bonus
    presence_bonus = float(np.interp(v_ratio, [0.30, 0.55, 0.75, 1.0], [0, 3, 5, 8]))

    raw = step_score + regularity_bonus + presence_bonus
    final = int(np.clip(raw, 60, 98))

    print(f"  [Skala] Steps={num_steps}, StepScore={step_score:.1f}, RegularityBonus={regularity_bonus:.1f}, PresenceBonus={presence_bonus:.1f}")
    print(f"  [Skala] 🎯 Markah Akhir: {final}%")
    return final


# ==============================================================
# BAHAGIAN 4: API ENDPOINT PENILAIAN LATIHAN
# ==============================================================
@app.route('/api/evaluate_exercise', methods=['POST'])
def evaluate_exercise():
    if 'audio' not in request.files:
        return jsonify({"error": "Tiada fail audio dijumpai"}), 400

    audio_file = request.files['audio']
    task_name = request.form.get('taskName', '')

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "temp_exercise.wav")
    audio_file.save(temp_path)

    try:
        y, sr = librosa.load(temp_path, sr=22050)  # Standardise sample rate

        # Trim leading/trailing silence but keep at least 0.5s
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(y_trimmed) > sr * 0.5:
            y = y_trimmed

        print(f"\n{'='*40}")
        print(f"🎙️ MENILAI TUGASAN: {task_name}")
        print(f"   Durasi audio: {len(y)/sr:.2f}s | Sample rate: {sr}Hz")
        print('='*40)

        # Route to appropriate scoring function based on exercise name
        if "Gumaman" in task_name or "Humming" in task_name:
            score = score_gumaman(y, sr)

        elif "Siren" in task_name:
            score = score_siren(y, sr)

        elif "Messa" in task_name:
            score = score_messa_di_voce(y, sr)

        elif "Skala" in task_name:
            score = score_skala_vokal(y, sr)

        else:
            # Unknown task: use generic pitch stability as fallback
            print(f"  [UNKNOWN TASK] Using generic stability scoring.")
            score = score_gumaman(y, sr)

        os.remove(temp_path)
        print(f"✅ Markah Dihantar: {score}%\n")
        return jsonify({"status": "success", "score": score})

    except Exception as e:
        print(f"❌ Ralat Penilaian DSP: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"status": "error", "score": 62})

# ==============================================================
# API PENGURUSAN PROFIL & DIAGNOSIS (MONGODB)
# ==============================================================
@app.route('/api/get_user', methods=['POST'])
def get_user():
    data = request.json
    username = data.get('username')
    
    # Cari pengguna, tapi jangan pulangkan kata laluan (0 bermaksud sembunyi)
    user = users_collection.find_one({"username": username}, {"_id": 0, "password": 0})
    
    if user:
        return jsonify({"status": "success", "data": user}), 200
    else:
        return jsonify({"error": "Pengguna tidak dijumpai"}), 404

@app.route('/api/update_diagnostic', methods=['POST'])
def update_diagnostic():
    data = request.json
    username = data.get('username')
    level = data.get('level')
    stage = data.get('stage', 1)

    if not username:
        return jsonify({"error": "Tiada nama pengguna"}), 400

    # Kemas kini status diagnosis pesakit ke dalam MongoDB
    users_collection.update_one(
        {"username": username},
        {"$set": {
            "diagnostic_status.level": level,
            "diagnostic_status.abnormalStage": stage,
            "diagnostic_status.tested": True
        }}
    )
    print(f"💾 Pangkalan Data Dikemas kini: {username} -> {level} (Tahap {stage})")
    
    return jsonify({"message": "Data berjaya disimpan!"}), 200

@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    data = request.json
    username = data.get('username')
    new_stage = data.get('stage')
    streak_stage = data.get('streak_stage') # Cth: 1, 2, atau 3
    streak_value = data.get('streak_value') # Cth: Berapa hari berturut-turut

    if not username:
        return jsonify({"error": "Tiada nama pengguna"}), 400

    # Sediakan kotak data apa yang perlu dikemas kini
    update_fields = {}
    
    if new_stage is not None:
        update_fields["diagnostic_status.abnormalStage"] = new_stage
        
    if streak_stage is not None and streak_value is not None:
        update_fields[f"rehab_progress.stage{streak_stage}_streak"] = streak_value

    # Jika ada data untuk dikemas kini, simpan ke MongoDB
    if update_fields:
        users_collection.update_one(
            {"username": username},
            {"$set": update_fields}
        )
        print(f"💾 Kemajuan Disimpan: {username} -> Peringkat {new_stage} | Streak Tahap {streak_stage}: {streak_value}")
    
    return jsonify({"message": "Kemajuan berjaya disimpan!"}), 200

if __name__ == '__main__':
    print("🚀 Pelayan AI sedia menerima sambungan di http://127.0.0.1:5000")
    print("   Algorithms: PYIN F0 | Stability | Glide Shape | Messa Shape | Scale Steps")
    app.run(debug=True, port=5000)