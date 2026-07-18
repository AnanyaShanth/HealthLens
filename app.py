import os, json, hashlib, requests
from flask import Flask, request, render_template
import easyocr
import google.generativeai as genai
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from gtts import gTTS

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = os.path.join('static', 'audio')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
reader = easyocr.Reader(['en'])

PROMPT_TEMPLATE = """
You are explaining medicine to someone who cannot read well and never studied science.
Explain like a very short, simple story or example, the way you'd tell a child or a
grandmother. Use everyday comparisons (soldiers fighting germs, ants at a picnic, watering
a plant, putting out a fire) instead of medical words.

Here are examples of the QUALITY and STYLE expected:

Example 1 - Amoxicillin (antibiotic):
- how_it_works: "Tiny germs are hiding inside you like ants in your kitchen. This medicine
  is like ant poison that goes to every corner and kills them, even the ones hiding deep."
- importance: "If you stop after just 3 days because you feel better, some strong ants are
  still alive and hiding. They will multiply again in a few days, and next time the same
  poison may not work on them - you could get sicker than before, and need a stronger,
  costlier medicine to fight them."

Example 2 - Paracetamol (painkiller/fever):
- how_it_works: "When you have fever, your body is like a room that got too hot. This
  medicine is like opening a window - it slowly cools the room down so you feel comfortable
  again."
- importance: "If you skip a dose while the fever is still high, the room heats up again
  and you may feel weak, shivery, or dizzy. Taking it on time keeps the temperature steady
  so your body can rest and heal."

Example 3 - Blood pressure tablet (long-term/chronic):
- how_it_works: "Your blood moves through pipes inside you. When the pressure is too high,
  it is like water pushed too hard through a garden hose - it can wear out the hose over
  time. This tablet gently loosens the flow so the pipes are not under strain."
- importance: "You may feel fine even without taking it - that is the danger. The pressure
  is still high inside, quietly damaging your heart, eyes, and kidneys, like a slow leak
  you cannot see. Skipping doses does not show pain right away, but the harm builds up
  silently over months and years."

Now follow this SAME style - a real, specific, sensory example (not vague words like
"germs will grow" or "it may not work") - for each medicine below.

Extracted prescription text (may contain OCR errors):
{ocr_text}

For EACH medicine, fill in these fields simply:
- name
- purpose: one simple sentence - what problem this fixes
- how_it_works: a tiny story/example (2 short sentences) of what the medicine does inside
  the body - use a simple everyday comparison, no medical jargon
- dosage
- timing: only "morning", "afternoon", "evening", or "night"
- how_to_take: with food / empty stomach / with water etc
- duration
- importance: a tiny story/example (2-3 short sentences) that explains the SPECIFIC,
  realistic consequence of skipping a dose or stopping early for THIS TYPE of medicine
  (e.g. antibiotics -> germs come back stronger and resistant; painkillers -> symptom
  returns and gets harder to control; chronic disease tablets -> silent long-term organ
  damage; missed insulin/diabetes -> sugar spikes and dizziness/fainting). Be concrete
  about what the person will actually feel or risk, not just "it won't work."
- precautions

Respond ONLY with valid JSON, no other text, in this exact format:
{{
  "medicines": [
    {{
      "name": "...", "purpose": "...", "how_it_works": "...", "dosage": "...",
      "timing": "...", "how_to_take": "...", "duration": "...", "importance": "...",
      "precautions": "..."
    }}
  ]
}}
"""

FIELDS = ['name', 'purpose', 'how_it_works', 'dosage', 'timing', 'how_to_take', 'duration', 'importance', 'precautions']

PERIOD_INFO = {
    'morning':   {'emoji': '☀️', 'label_en': 'Morning',   'label_ml': 'രാവിലെ',      'color': '#e8a33d', 'css': 'morning'},
    'afternoon': {'emoji': '🌤️', 'label_en': 'Afternoon', 'label_ml': 'ഉച്ചയ്ക്ക്',   'color': '#e86a33', 'css': 'afternoon'},
    'evening':   {'emoji': '🌇', 'label_en': 'Evening',   'label_ml': 'വൈകുന്നേരം',  'color': '#b23a62', 'css': 'evening'},
    'night':     {'emoji': '🌙', 'label_en': 'Night',     'label_ml': 'രാത്രി',      'color': '#33437a', 'css': 'night'},
}
DEFAULT_BADGE = {'emoji': '💊', 'label_en': 'As directed', 'label_ml': 'നിർദ്ദേശ പ്രകാരം', 'color': '#1f5c4e', 'css': 'default'}


def get_timing_badges(timing_text):
    t = (timing_text or '').lower()
    badges = []
    if 'morning' in t: badges.append(PERIOD_INFO['morning'])
    if 'afternoon' in t or 'noon' in t: badges.append(PERIOD_INFO['afternoon'])
    if 'evening' in t: badges.append(PERIOD_INFO['evening'])
    if 'night' in t: badges.append(PERIOD_INFO['night'])
    return badges if badges else [DEFAULT_BADGE]


def get_intake_icon(text):
    t = (text or '').lower()
    if 'empty stomach' in t: return '⛔🍽️'
    if 'food' in t or 'meal' in t: return '🍽️'
    if 'water' in t: return '💧'
    return '💊'


def call_gemma(prompt):
    resp = requests.post(
        'http://localhost:11434/api/generate',
        json={"model": "gemma2:2b", "prompt": prompt, "stream": False},
        timeout=90
    )
    resp.raise_for_status()
    return resp.json()['response']


def call_gemini(prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    return model.generate_content(prompt).text


def translate_to_malayalam(text):
    try:
        return GoogleTranslator(source='en', target='ml').translate(text)
    except Exception:
        return "(translation unavailable)"


def get_or_create_audio(text, lang='ml'):
    """Generate TTS audio once, cache by content hash so repeat text doesn't regenerate.
    Returns a web URL (always forward slashes) or None if generation fails."""
    if not text or not text.strip():
        return None
    key = hashlib.md5(f"{lang}:{text}".encode('utf-8')).hexdigest()
    filename = f"{key}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(filepath)
        except Exception as e:
            print(f"TTS failed: {e}")
            return None
    # Build URL manually with forward slashes - os.path.join uses backslashes on Windows,
    # which breaks the URL when rendered in HTML.
    return f"/static/audio/{filename}"


def build_full_narration_en(med):
    return (
        f"{med.get('name', '')}. "
        f"Purpose: {med.get('purpose', '')}. "
        f"{med.get('how_it_works', '')}. "
        f"Dosage: {med.get('dosage', '')}. "
        f"Take in the {med.get('timing', '')}. "
        f"{med.get('how_to_take', '')}. "
        f"Duration: {med.get('duration', '')}. "
        f"Important: {med.get('importance', '')}. "
        f"Precautions: {med.get('precautions', '')}."
    )


def build_full_narration_ml(med):
    return (
        f"{med.get('name_ml', '')}. "
        f"ഉപയോഗം: {med.get('purpose_ml', '')}. "
        f"{med.get('how_it_works_ml', '')}. "
        f"ഡോസ്: {med.get('dosage_ml', '')}. "
        f"{med.get('timing_ml', '')} കഴിക്കുക. "
        f"{med.get('how_to_take_ml', '')}. "
        f"ദൈർഘ്യം: {med.get('duration_ml', '')}. "
        f"പ്രധാനം: {med.get('importance_ml', '')}. "
        f"മുൻകരുതലുകൾ: {med.get('precautions_ml', '')}."
    )


def get_ai_analysis(ocr_text):
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)
    try:
        raw = call_gemma(prompt)
        source = "gemma"
    except Exception as e:
        print(f"Gemma unavailable ({e}), falling back to Gemini")
        raw = call_gemini(prompt)
        source = "gemini"

    cleaned = raw.strip().replace('```json', '').replace('```', '').strip()
    try:
        data = json.loads(cleaned)
        if not data.get('medicines'):
            data = {"error": "Couldn't identify any medicines in this prescription. Try a clearer photo."}
        else:
            for med in data.get('medicines', []):
                badges = get_timing_badges(med.get('timing', ''))
                med['timing_badges'] = badges
                med['accent_color'] = badges[0]['color']
                med['intake_icon'] = get_intake_icon(med.get('how_to_take', ''))
                for field in FIELDS:
                    med[f'{field}_ml'] = translate_to_malayalam(med.get(field, ''))

                # Full-card read-aloud, one audio file per language per medicine
                full_text_en = build_full_narration_en(med)
                full_text_ml = build_full_narration_ml(med)
                med['audio_full_en'] = get_or_create_audio(full_text_en, 'en')
                med['audio_full_ml'] = get_or_create_audio(full_text_ml, 'ml')
    except Exception:
        data = {"error": "Could not parse response", "raw": cleaned}

    data['_source'] = source
    return data


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('prescription')
    if not file or file.filename == '':
        return render_template('result.html', data={"error": "No file uploaded. Please choose an image."})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        ocr_text = " ".join(reader.readtext(filepath, detail=0))
        if not ocr_text.strip():
            return render_template('result.html', data={"error": "Couldn't read any text from this image. Try a clearer photo."})
        data = get_ai_analysis(ocr_text)
        data['_ocr_text'] = ocr_text
    except Exception as e:
        data = {"error": f"Something went wrong: {e}"}

    return render_template('result.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)