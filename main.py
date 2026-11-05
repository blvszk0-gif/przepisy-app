import os
import json
import tempfile
import glob
import time
from fastapi import FastAPI, Request, Form
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

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    recipes = []
    if SUPABASE_URL:
        try:
            response = supabase.table("recipes").select("*").order("created_at", desc=True).execute()
            recipes = response.data
        except Exception as e:
            print(f"Błąd bazy danych: {e}")
    return templates.TemplateResponse(request=request, name="index.html", context={"recipes": recipes})

@app.post("/process")
def process_reels(url: str = Form(...)):
    try:
        description = ""
        transcript_text = ""

        # 1. Pobieranie danych z Instagrama (wymuszenie strumienia audio)
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'ba/b',  # Pobiera sam dźwięk lub plik ze ścieżką audio
                'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                description = info.get('description', '')

            # Znalezienie pobranego pliku w folderze tymczasowym
            downloaded_files = glob.glob(f"{temp_dir}/*")

            # 2. Bezpieczna transkrypcja z Groq (jeśli plik zawiera dźwięk)
            if downloaded_files and GROQ_API_KEY:
                media_file = downloaded_files[0]
                try:
                    groq_client = Groq(api_key=GROQ_API_KEY)
                    with open(media_file, "rb") as file:
                        transcription = groq_client.audio.transcriptions.create(
                            file=(os.path.basename(media_file), file.read()),
                            model="whisper-large-v3"
                        )
                    transcript_text = transcription.text
                except Exception as audio_err:
                    print(f"Pominięto transkrypcję (brak wykrywalnego dźwięku/mowy): {audio_err}")
                    transcript_text = ""

        # 3. Analiza i tłumaczenie przez Gemini API (z automatycznym ponawianiem przy 503)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        Przeanalizuj przepis kulinarny.
        Opis pod postem: {description}
        Transkrypcja wideo (mowa): {transcript_text}
        
        Wymagania:
        1. Jeśli w opisie lub transkrypcji znajduje się przepis, ułóż z niego czytelną listę składników i kroków.
        2. Przetłumacz na język polski. Przelicz jednostki (np. uncje na gramy).
        3. Zwróć wynik jako czysty JSON o strukturze:
        {{
            "title": "Krótki tytuł przepisu",
            "ingredients": [{{"name": "nazwa produktu i ilość", "category": "dział w sklepie (np. Nabiał, Warzywa, Mięso)"}}],
            "instructions": ["krok 1", "krok 2"]
        }}
        """
        
        ai_response = None
        # Próbujemy do 3 razy z krótkim odstępem czasowym
        for attempt in range(3):
            try:
                ai_response = gemini_client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(3)  # odczekaj 3 sekundy, aż obciążenie serwera spadnie
                    continue
                raise e

        if not ai_response:
            raise Exception("Serwery AI były chwilowo zajęte. Spróbuj kliknąć ponownie za chwilę.")
        
        recipe_data = json.loads(ai_response.text)

        # 4. Zapis do Supabase
        if SUPABASE_URL:
            supabase.table("recipes").insert({
                "title": recipe_data["title"],
                "ingredients": recipe_data["ingredients"],
                "instructions": recipe_data["instructions"],
                "source_url": url
            }).execute()
            
        return {"status": "success", "data": recipe_data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
