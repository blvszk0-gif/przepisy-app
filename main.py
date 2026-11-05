import os
import json
import tempfile
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

# Konfiguracja API i Bazy Danych pobierana ze zmiennych środowiskowych
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Inicjalizacja klientów (tylko jeśli klucze są dostępne)
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # Pobieranie zapisanych przepisów z bazy danych przy otwarciu strony
    recipes = []
    if SUPABASE_URL:
        response = supabase.table("recipes").select("*").order("created_at", desc=True).execute()
        recipes = response.data
    return templates.TemplateResponse(request=request, name="index.html", context={"recipes": recipes})

@app.post("/process")
def process_reels(url: str = Form(...)):
    try:
        # 1. Pobieranie z Instagrama (wybieramy najmniejszy plik wideo, by oszczędzić czas i RAM)
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'worst[ext=mp4]',
                'outtmpl': f'{temp_dir}/%(id)s.%(ext)s',
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                description = info.get('description', '')
                video_filename = ydl.prepare_filename(info)

            # 2. Transkrypcja wideo przez Groq (Whisper)
            groq_client = Groq(api_key=GROQ_API_KEY)
            with open(video_filename, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(video_filename), file.read()),
                    model="whisper-large-v3"
                )
            transcript_text = transcription.text

        # 3. Analiza i tłumaczenie przez Gemini API
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""
        Przeanalizuj przepis kulinarny.
        Opis pod postem: {description}
        Transkrypcja wideo (mowa): {transcript_text}
        
        Wymagania:
        1. Przetłumacz na język polski. Przelicz jednostki (np. uncje na gramy).
        2. Zwróć wynik jako czysty JSON o strukturze:
        {{
            "title": "Krótki tytuł przepisu",
            "ingredients": [{{"name": "nazwa produktu i ilość", "category": "dział w sklepie (np. Nabiał, Warzywa)"}}],
            "instructions": ["krok 1", "krok 2"]
        }}
        """
        
        ai_response = gemini_client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        recipe_data = json.loads(ai_response.text)

        # 4. Zapis do bazy danych Supabase
        if SUPABASE_URL:
            db_insert = supabase.table("recipes").insert({
                "title": recipe_data["title"],
                "ingredients": recipe_data["ingredients"],
                "instructions": recipe_data["instructions"],
                "source_url": url
            }).execute()
            
        return {"status": "success", "data": recipe_data}

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
