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

def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini_with_retry(prompt: str, max_retries: int = 3):
    client = get_gemini_client()
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
                time.sleep(3)
                continue
            raise e

def process_reels_task(url: str):
    """Zadanie wykonywane w tle – bezpieczne, z limitami czasowymi."""
    try:
        description = ""
        # ETAP 1: Błyskawiczne pobranie samych metadanych (opisu) bez ściągania wideo
        ydl_opts_meta = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 15,
        }
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '') or ''

        # Sprawdzamy, czy w opisie jest już kompletny przepis
        recipe_data = None
        if len(description.strip()) > 40:
            prompt_check = f"""
            Przeanalizuj opis z rolki na Instagramie:
            \"\"\"{description}\"\"\"

            Czy ten opis zawiera składniki ORAZ kroki przygotowania dania?
            Jeśli TAK, ułóż z niego przepis, przetłumacz na polski i zwróć JSON:
            {{
                "has_recipe": true,
                "title": "Tytuł dania",
                "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie"}}],
                "instructions": ["krok 1", "krok 2"]
            }}
            Jeśli NIE (np. brak składników lub tylko krótki opis bez proporcji), zwróć po prostu:
            {{"has_recipe": false}}
            """
            try:
                result = ask_gemini_with_retry(prompt_check)
                if result.get("has_recipe"):
                    recipe_data = result
            except Exception as e:
                print(f"Błąd analizy opisu: {e}")

        # ETAP 2: Jeśli w opisie nie ma przepisu, dopiero wtedy pobieramy dźwięk i robimy transkrypcję
        if not recipe_data:
            transcript_text = ""
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts_audio = {
                    'format': 'ba/b',
                    'outtmpl': f'{temp_dir}/audio.%(ext)s',
                    'quiet': True,
                    'no_warnings': True,
                    'socket_timeout': 20,
                    'max_filesize': 35 * 1024 * 1024,
                }
                with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                    ydl.download([url])

                downloaded_files = glob.glob(f"{temp_dir}/*")
                if downloaded_files and GROQ_API_KEY:
                    media_file = downloaded_files[0]
                    try:
                        groq_client = Groq(api_key=GROQ_API_KEY)
                        with open(media_file, "rb") as f:
                            transcription = groq_client.audio.transcriptions.create(
                                file=(os.path.basename(media_file), f.read()),
                                model="whisper-large-v3"
                            )
                        transcript_text = transcription.text
                    except Exception as audio_err:
                        print(f"Błąd transkrypcji: {audio_err}")

            # Wysyłamy połączony opis i transkrypcję do Gemini
            prompt_full = f"""
            Stwórz przepis kulinarny na podstawie danych:
            Opis: {description}
            Transkrypcja audio: {transcript_text}

            Przetłumacz na polski i zwróć JSON:
            {{
                "title": "Tytuł dania",
                "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie"}}],
                "instructions": ["krok 1", "krok 2"]
            }}
            """
            recipe_data = ask_gemini_with_retry(prompt_full)

        # ETAP 3: Zapis do bazy danych Supabase
        if recipe_data and supabase:
            supabase.table("recipes").insert({
                "title": recipe_data.get("title", "Przepis z Reels"),
                "ingredients": recipe_data.get("ingredients", []),
                "instructions": recipe_data.get("instructions", []),
                "source_url": url
            }).execute()
            print(f"Pomyślnie dodano przepis: {recipe_data.get('title')}")

    except Exception as err:
        print(f"Błąd przetwarzania w tle dla {url}: {err}")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    recipes = []
    if supabase:
        try:
            response = supabase.table("recipes").select("*").order("created_at", desc=True).execute()
            recipes = response.data
        except Exception as e:
            print(f"Błąd pobierania bazy: {e}")
    return templates.TemplateResponse(request=request, name="index.html", context={"recipes": recipes})

@app.get("/api/recipes")
def get_recipes_api():
    """Endpoint do cichego sprawdzania nowo dodanych przepisów w tle."""
    if supabase:
        res = supabase.table("recipes").select("id, title, created_at").order("created_at", desc=True).execute()
        return res.data
    return []

@app.post("/process")
def process_reels(background_tasks: BackgroundTasks, url: str = Form(...)):
    # Natychmiast dodajemy zadanie do kolejki w tle – odpowiedź wraca w 0.2 sekundy!
    background_tasks.add_task(process_reels_task, url.strip())
    return {"status": "queued", "message": "Rolka dodana do kolejki przetwarzania!"}
