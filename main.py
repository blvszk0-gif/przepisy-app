import os
import json
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

def ask_gemini_with_retry(prompt: str, max_retries: int = 4):
    client = genai.Client(api_key=GEMINI_API_KEY)
    delays = [2, 5, 10, 15]
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < max_retries - 1:
                time.sleep(delays[attempt])
                continue
            raise e

def process_recipe_by_id(recipe_id: str, url: str):
    try:
        # KROK 1: Pobieranie opisu posta (bez pobierania wideo - zawsze działa w 100%)
        supabase.table("recipes").update({
            "status": "processing",
            "progress_step": 1,
            "progress_text": "Pobieranie opisu posta i multimediów..."
        }).eq("id", recipe_id).execute()

        description = ""
        transcript_text = ""

        # 1A. Bezpieczne wyciągnięcie tekstu opisu
        with yt_dlp.YoutubeDL({'skip_download': True, 'quiet': True, 'no_warnings': True, 'socket_timeout': 15}) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '') or ''

        # Zapisujemy opis w bazie już teraz, by nie przepadł
        supabase.table("recipes").update({
            "original_description": description,
            "progress_step": 2,
            "progress_text": "Pobieranie i transkrypcja mowy (Whisper)..."
        }).eq("id", recipe_id).execute()

        # 1B. Bezpieczne pobranie strumienia multimedialnego (BEZ postprocessorów ffprobe)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Wybieramy najlżejszy strumień zawierający dźwięk lub małe wideo
                ydl_opts = {
                    'format': 'ba/b[height<=480]/b/worst',
                    'outtmpl': f'{temp_dir}/media.%(ext)s',
                    'quiet': True,
                    'no_warnings': True,
                    'socket_timeout': 30,
                    'max_filesize': 30 * 1024 * 1024,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                downloaded_files = glob.glob(f"{temp_dir}/*")
                if downloaded_files and GROQ_API_KEY:
                    media_file = downloaded_files[0]
                    # Groq natywnie obsługuje mp4, m4a, mp3 bez potrzeby konwersji
                    groq_client = Groq(api_key=GROQ_API_KEY)
                    with open(media_file, "rb") as f:
                        transcription = groq_client.audio.transcriptions.create(
                            file=(os.path.basename(media_file), f.read()),
                            model="whisper-large-v3"
                        )
                    transcript_text = transcription.text.strip()
                    print(f"Pomyślna transkrypcja: {transcript_text[:50]}...")
        except Exception as audio_err:
            print(f"Pominięto audio (błąd odczytu lub brak mowy): {audio_err}")
            transcript_text = "(Nie wykryto mowy w filmie lub audio niedostępne)"

        # KROK 3: Przetworzenie całości przez Gemini
        supabase.table("recipes").update({
            "progress_step": 3,
            "progress_text": "Scalanie opisu i transkrypcji przez Gemini...",
            "transcript": transcript_text
        }).eq("id", recipe_id).execute()

        prompt = f"""
        Twoim zadaniem jest stworzyć kompletny przepis kulinarny łącząc DWA źródła:
        
        1. OPIS Z POSTA NA INSTAGRAMIE:
        \"\"\"{description}\"\"\"

        2. TRANSKRYPCJA MOWY Z WIDEO (Whisper):
        \"\"\"{transcript_text}\"\"\"

        Wytyczne:
        - Jeśli opis zawiera składniki, a na wideo twórca podaje instrukcje lub odwrotnie – połącz wszystko w jedną spójną całość.
        - Jeśli mowa jest niedostępna, opieraj się w 100% na opisie posta.
        - Przetłumacz na język polski. Przelicz jednostki (uncje, cup na gramy/ml/łyżki).
        - Zwróć wynik jako JSON:
        {{
            "title": "Tytuł dania",
            "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie (np. Warzywa, Nabiał, Przyprawy)"}}],
            "instructions": ["krok 1", "krok 2"]
        }}
        """

        recipe_data = ask_gemini_with_retry(prompt)

        # KROK 4: Zapis gotowego dania w Supabase
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
