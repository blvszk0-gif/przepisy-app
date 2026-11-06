import os
import json
import re
import tempfile
import glob
import time
from typing import List
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import yt_dlp
from groq import Groq
from google import genai
from google.genai import types
from supabase import create_client, Client
from pydantic import BaseModel, Field

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# MODEL STRUKTURY DANYCH (PYDANTIC)
# ==========================================
class IngredientModel(BaseModel):
    name: str = Field(description="Dokładna nazwa składnika wraz z ilością, np. '3 jajka', '150 ml śmietanki 30%', '180 g herbatników'")
    category: str = Field(description="Kategoria sklepowa, np. Nabiał, Warzywa i owoce, Produkty sypkie, Mięso, Inne")

class RecipeResponseModel(BaseModel):
    title: str = Field(description="Konkretna nazwa potrawy, np. Kremowy sernik, Puszysty omlet")
    ingredients: List[IngredientModel] = Field(description="Lista wszystkich realnych składników z przepisu")
    instructions: List[str] = Field(description="Kroki przygotowania krok po kroku")


def ask_ai_structured(description: str, transcript_text: str):
    """Używa Gemini z wymuszonym schematem Pydantic i automatycznym ponawianiem przy przekroczeniu limitu (429)."""
    if not GEMINI_API_KEY:
        raise Exception("Brak klucza GEMINI_API_KEY w zmiennych środowiskowych!")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    Przeanalizuj poniższy opis z Instagrama oraz transkrypcję audio i wyciągnij z nich DOKŁADNY przepis kulinarny.
    Bezwzględnie ignoruj hasztagi, teksty sponsorowane, prośby o obserwowanie oraz zwroty marketingowe. Nie wymyślaj składników, których nie ma w tekście.

    OPIS Z INSTAGRAMA:
    \"\"\"{description}\"\"\"

    TRANSKRYPCJA AUDIO:
    \"\"\"{transcript_text}\"\"\"
    """

    max_retries = 5
    backoff_factor = 5  # Liczba sekund bazowych do odczekania

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RecipeResponseModel,
                    temperature=0.0,
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            err_msg = str(e)
            # Sprawdzamy czy to błąd przekroczenia limitu (429 / RESOURCE_EXHAUSTED) lub przeciążenie (503)
            if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries - 1:
                sleep_time = backoff_factor * (attempt + 1)  # Rosnący czas oczekiwania: 5s, 10s, 15s...
                print(f"Przekroczono limit zapytań (429). Czekam {sleep_time} sekund przed ponowieniem (próba {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
                continue
            if attempt == max_retries - 1:
                raise e
    raise Exception("Brak odpowiedzi od Gemini API po wielokrotnych próbach")


def sanitize_recipe_data(raw_data: dict, fallback_description: str = "") -> dict:
    """Weryfikuje i oczyszcza dane w razie awaryjnych sytuacji."""
    raw_title = str(raw_data.get("title", "")).strip()
    
    if not raw_title or "apetyczny tytuł" in raw_title.lower() or "przepis z reels" in raw_title.lower():
        lines = [l.strip() for l in fallback_description.split("\n") if len(l.strip()) > 3]
        raw_title = lines[0][:60] if lines else "Przepis z Instagrama"

    dummy_phrases = [
        "dokładna nazwa", "kategoria w markecie", "ilość produktu", 
        "przygotowanie...", "krok 1:", "krok 2:", "kombinuj składniki"
    ]

    cleaned_ingredients = []
    seen = set()
    banned_words = {"składniki", "składnik", "ingredients", "ingredient", "żywność"}

    for item in raw_data.get("ingredients", []):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            cat = str(item.get("category", "Inne")).strip() or "Inne"
        else:
            name = str(item).strip()
            cat = "Inne"

        name_lower = name.lower()
        if any(d in name_lower for d in dummy_phrases):
            continue
        if not name or len(name) < 2 or name_lower in banned_words or name_lower in seen:
            continue
        seen.add(name_lower)
        cleaned_ingredients.append({"name": name, "category": cat})

    cleaned_instructions = []
    seen_steps = set()

    for step in raw_data.get("instructions", []):
        s = str(step).strip()
        s_lower = s.lower()
        if any(d in s_lower for d in dummy_phrases):
            continue
        if s and len(s) > 3 and s_lower not in seen_steps:
            seen_steps.add(s_lower)
            cleaned_instructions.append(s)

    return {
        "title": raw_title,
        "ingredients": cleaned_ingredients,
        "instructions": cleaned_instructions
    }


def process_recipe_by_id(recipe_id: str, url: str):
    try:
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
                                model="whisper-large-v3",
                                language="pl",
                                prompt="Kulinaria, gotowanie, pieczenie, składniki, gramy, łyżki, proporcje."
                            )
                        transcript_text = transcription.text.strip()
            except Exception:
                transcript_text = "(Nie wykryto mowy w filmie)"
        else:
            transcript_text = "(Rolka bez ścieżki dźwiękowej)"

        supabase.table("recipes").update({
            "progress_step": 3,
            "progress_text": "Analiza treści przez Gemini...",
            "transcript": transcript_text
        }).eq("id", recipe_id).execute()

        raw_recipe = ask_ai_structured(description, transcript_text)
        recipe_data = sanitize_recipe_data(raw_recipe, fallback_description=description)

        supabase.table("recipes").update({
            "title": recipe_data["title"],
            "ingredients": recipe_data["ingredients"],
            "instructions": recipe_data["instructions"],
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


@app.post("/edit/{recipe_id}")
def edit_recipe(
    recipe_id: str,
    title: str = Form(...),
    ingredients_text: str = Form(...),
    instructions_text: str = Form(...)
):
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})

    ingredients = []
    for line in ingredients_text.strip().split("\n"):
        line = line.strip().lstrip("-*•0123456789. ")
        if not line:
            continue
        cat = "Inne"
        match = re.search(r'\[(.*?)\]', line)
        if match:
            cat = match.group(1).strip()
            line = re.sub(r'\[.*?\]', '', line).strip()
        ingredients.append({"name": line, "category": cat})

    instructions = [
        line.strip().lstrip("-*•0123456789. ")
        for line in instructions_text.strip().split("\n")
        if line.strip()
    ]

    try:
        supabase.table("recipes").update({
            "title": title.strip(),
            "ingredients": ingredients,
            "instructions": instructions
        }).eq("id", recipe_id).execute()
        return {"status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


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