import os
import json
import tempfile
import glob
import time
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import yt_dlp
import static_ffmpeg
from groq import Groq
from google import genai
from google.genai import types
from supabase import create_client, Client

# Dodanie pełnego ffmpeg oraz ffprobe do systemowego PATH
static_ffmpeg.add_paths()

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
        # KROK 1: Pobieranie opisu oraz konwersja do MP3
        supabase.table("recipes").update({
            "status": "processing",
            "progress_step": 1,
            "progress_text": "Pobieranie opisu i wycinanie ścieżki audio MP3..."
        }).eq("id", recipe_id).execute()

        description = ""
        transcript_text = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
                'outtmpl': f'{temp_dir}/audio.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                'socket_timeout': 30,
                'max_filesize': 50 * 1024 * 1024,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                description = info.get('description', '') or ''

            # KROK 2: Transkrypcja mowy twórcy
            supabase.table("recipes").update({
                "progress_step": 2,
                "progress_text": "Transkrypcja mowy twórcy przez Whisper...",
                "original_description": description
            }).eq("id", recipe_id).execute()

            mp3_files = glob.glob(f"{temp_dir}/*.mp3")
            media_file = mp3_files[0] if mp3_files else (glob.glob(f"{temp_dir}/*") or [None])[0]

            if media_file and GROQ_API_KEY:
                try:
                    groq_client = Groq(api_key=GROQ_API_KEY)
                    with open(media_file, "rb") as f:
                        transcription = groq_client.audio.transcriptions.create(
                            file=(os.path.basename(media_file), f.read()),
                            model="whisper-large-v3"
                        )
                    transcript_text = transcription.text
                except Exception as audio_err:
                    print(f"Błąd transkrypcji Groq: {audio_err}")
                    transcript_text = "(Nie wykryto mowy w filmie)"

        # KROK 3: Łączenie opisu i transkrypcji przez Gemini
        supabase.table("recipes").update({
            "progress_step": 3,
            "progress_text": "Scalanie opisu posta i transkrypcji audio...",
            "transcript": transcript_text
        }).eq("id", recipe_id).execute()

        prompt = f"""
        Twoim zadaniem jest stworzyć kompletny przepis kulinarny łącząc DWA źródła:
        
        1. OPIS Z POSTA NA INSTAGRAMIE:
        \"\"\"{description}\"\"\"

        2. TRANSKRYPCJA MOWY Z WIDEO (Whisper):
        \"\"\"{transcript_text}\"\"\"

        Wytyczne:
        - Połącz składniki i wskazówki z obu źródeł w jeden pełny przepis. Jeśli nie ma mowy, oprzyj się w 100% na opisie.
        - Przetłumacz na język polski. Przelicz jednostki (uncje, cup na gramy/ml/łyżki).
        - Zwróć wynik jako JSON:
        {{
            "title": "Tytuł dania",
            "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie (np. Warzywa, Nabiał, Przyprawy)"}}],
            "instructions": ["krok 1", "krok 2"]
        }}
        """

        recipe_data = ask_gemini_with_retry(prompt)

        # KROK 4: Zapis gotowego przepisu
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
