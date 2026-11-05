import os
import json
import re
import tempfile
import glob
import time
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import yt_dlp
from groq import Groq
from google import genai
from google.genai import types
from supabase import create_client, Client

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def ask_groq_ai(prompt: str):
    """Zapasowy silnik AI pobierający wyłącznie aktywne modele z konta Groq."""
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # 1. Pobranie aktualnie aktywnych modeli tekstowych
    active_models = []
    try:
        models_data = groq_client.models.list().data
        for m in models_data:
            mid = m.id.lower()
            is_active = getattr(m, 'active', True)
            is_specialized = any(x in mid for x in ["whisper", "guard", "orpheus", "embed", "tts", "vision", "safetensors"])
            if is_active and not is_specialized and "/" not in m.id:
                active_models.append(m.id)
    except Exception as e:
        print(f"Błąd sprawdzania modeli Groq: {e}")

    if not active_models:
        active_models = ["llama-3.1-8b-instant"]

    # Priorytetyzacja sprawdzonych modeli Llama
    active_models.sort(key=lambda name: 0 if "llama-3.1-8b" in name else (1 if "llama-3.3" in name else 2))

    for model_name in active_models:
        # Próba 1: Tryb wymuszonego JSON
        try:
            print(f"Uruchamiam model Groq: {model_name}")
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Jesteś kucharzem i tłumaczem. Zwracasz WYŁĄCZNIE poprawny JSON bez żadnego dodatkowego tekstu."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2048
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e1:
            print(f"Model {model_name} (JSON mode) błąd: {e1}. Próba w trybie swobodnym...")
            
            # Próba 2: Tryb swobodny z ekstrakcją klamer regexem
            try:
                resp = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Zwróć przepis jako obiekt JSON: {\"title\": \"...\", \"ingredients\": [{\"name\": \"...\", \"category\": \"...\"}], \"instructions\": [\"...\"]}."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                raw_text = resp.choices[0].message.content.strip()
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception as e2:
                print(f"Model {model_name} (tekst) błąd: {e2}")
                continue

    raise Exception("Żaden z aktywnych modeli Groq nie przetworzył zapytania.")

def ask_ai_smart(prompt: str):
    """Odpytuje Gemini (z ponawianiem przy 503), a w razie braku limitu przełącza na Groq."""
    if GEMINI_API_KEY:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(3):
            try:
                res = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                return json.loads(res.text)
            except Exception as e:
                err_msg = str(e)
                if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < 2:
                    print(f"[Gemini 503] Ponawiam próbę za 3s ({attempt+1}/3)...")
                    time.sleep(3)
                    continue
                print(f"[Gemini błąd/limit]: {err_msg[:90]}. Przełączam na Groq...")
                break

    return ask_groq_ai(prompt)

def process_recipe_by_id(recipe_id: str, url: str):
    try:
        # KROK 1: Pobranie opisu i formatów
        supabase.table("recipes").update({
            "status": "processing",
            "progress_step": 1,
            "progress_text": "Pobieranie opisu posta i wideo..."
        }).eq("id", recipe_id).execute()

        description = ""
        transcript_text = ""
        formats = []

        with yt_dlp.YoutubeDL({'skip_download': True, 'quiet': True, 'no_warnings': True, 'socket_timeout': 15}) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '') or ''
            formats = info.get('formats', []) or []

        supabase.table("recipes").update({
            "original_description": description,
            "progress_step": 2,
            "progress_text": "Pobieranie audio i transkrypcja mowy (Whisper)..."
        }).eq("id", recipe_id).execute()

        # KROK 2: Weryfikacja obecności ścieżki dźwiękowej
        audio_stream_id = None
        for f in formats:
            if f.get('acodec') and f.get('acodec') != 'none':
                audio_stream_id = str(f.get('format_id'))
                break

        if audio_stream_id and GROQ_API_KEY:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    ydl_opts = {
                        'format': audio_stream_id,
                        'outtmpl': f'{temp_dir}/media.%(ext)s',
                        'quiet': True,
                        'no_warnings': True,
                        'socket_timeout': 30,
                        'max_filesize': 35 * 1024 * 1024,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])

                    downloaded_files = glob.glob(f"{temp_dir}/*")
                    if downloaded_files:
                        media_file = downloaded_files[0]
                        groq_client = Groq(api_key=GROQ_API_KEY)
                        with open(media_file, "rb") as f:
                            transcription = groq_client.audio.transcriptions.create(
                                file=(os.path.basename(media_file), f.read()),
                                model="whisper-large-v3"
                            )
                        transcript_text = transcription.text.strip()
                        print(f"Transkrypcja Whisper gotowa ({len(transcript_text)} znaków)")
            except Exception as audio_err:
                print(f"Transkrypcja pominięta: {audio_err}")
                transcript_text = "(Nie wykryto mowy w filmie)"
        else:
            transcript_text = "(Rolka bez ścieżki dźwiękowej)"

        # KROK 3: Przetworzenie całości przez AI
        supabase.table("recipes").update({
            "progress_step": 3,
            "progress_text": "Scalanie opisu i transkrypcji przez AI...",
            "transcript": transcript_text
        }).eq("id", recipe_id).execute()

        prompt = f"""
        Stwórz kompletny, zwięzły przepis kulinarny w języku polskim na podstawie danych:

        ŹRÓDŁO 1 (Opis posta na Instagramie):
        \"\"\"{description}\"\"\"

        ŹRÓDŁO 2 (Mowa twórcy z nagrania):
        \"\"\"{transcript_text}\"\"\"

        Wytyczne:
        - Połącz składniki i proporcje z obu źródeł w spójną całość.
        - Przetłumacz na polski i przelicz jednostki (uncje, cup na gramy/ml/łyżki).
        - Zwróć wyłącznie JSON o strukturze:
        {{
            "title": "Dokładny tytuł dania po polsku",
            "ingredients": [{{"name": "nazwa i ilość produktu", "category": "dział w sklepie"}}],
            "instructions": ["Krok 1 przygotowania", "Krok 2 przygotowania"]
        }}
        """

        recipe_data = ask_ai_smart(prompt)

        # KROK 4: Zapis do Supabase
        supabase.table("recipes").update({
            "title": recipe_data.get("title", "Przepis z Reels"),
            "ingredients": recipe_data.get("ingredients", []),
            "instructions": recipe_data.get("instructions", []),
            "status": "completed",
            "progress_step": 4,
            "progress_text": "Ukończono!",
            "transcript": transcript_text,
            "original_description": description
        }).eq("id", recipe_id).execute()

    except Exception as err:
        print(f"Błąd dla przepisu {recipe_id}: {err}")
        if supabase:
            supabase.table("recipes").update({
                "status": "error",
                "progress_text": "Błąd przetwarzania",
                "error_message": str(err)
            }).eq("id", recipe_id).execute()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    recipes = []
    if supabase:
        try:
            res = supabase.table("recipes").select("*").eq("status", "completed").order("created_at", desc=True).execute()
            recipes = res.data or []
        except Exception as e:
            print(f"Błąd bazy: {e}")
    return templates.TemplateResponse(request=request, name="index.html", context={"recipes": recipes})

@app.get("/api/queue")
def get_queue_api():
    if supabase:
        res = supabase.table("recipes").select("*").neq("status", "completed").order("created_at", desc=True).execute()
        return res.data or []
    return []

@app.get("/api/recipes-count")
def get_recipes_count():
    if supabase:
        res = supabase.table("recipes").select("id", count="exact").eq("status", "completed").execute()
        return {"count": res.count or 0}
    return {"count": 0}

@app.post("/process")
def process_reels(background_tasks: BackgroundTasks, url: str = Form(...)):
    url = url.strip()
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})

    res = supabase.table("recipes").insert({
        "title": "Pobieranie w toku...",
        "source_url": url,
        "status": "queued",
        "progress_step": 1,
        "progress_text": "Oczekiwanie w kolejce...",
        "ingredients": [],
        "instructions": []
    }).execute()

    recipe_id = res.data[0]["id"]
    background_tasks.add_task(process_recipe_by_id, recipe_id, url)
    return {"status": "queued", "id": recipe_id}

@app.post("/retry/{recipe_id}")
def retry_recipe(recipe_id: str, background_tasks: BackgroundTasks):
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})

    res = supabase.table("recipes").select("*").eq("id", recipe_id).execute()
    if not res.data:
        return JSONResponse(status_code=404, content={"message": "Nie znaleziono"})

    recipe = res.data[0]
    supabase.table("recipes").update({
        "title": "Ponawianie...",
        "status": "queued",
        "progress_step": 1,
        "progress_text": "Restartowanie zadania...",
        "error_message": ""
    }).eq("id", recipe_id).execute()

    background_tasks.add_task(process_recipe_by_id, recipe_id, recipe["source_url"])
    return {"status": "retrying"}

@app.post("/delete/{recipe_id}")
def delete_recipe(recipe_id: str):
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})
    try:
        supabase.table("recipes").delete().eq("id", recipe_id).execute()
        return {"status": "deleted"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
