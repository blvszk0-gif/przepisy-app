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

def ask_gemini_with_retry(prompt: str, max_retries: int = 4):
    client = get_gemini_client()
    delays = [2, 6, 12, 20]
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return json.loads(response.text)
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries - 1:
                time.sleep(delays[attempt])
                continue
            raise e

def process_recipe_by_id(recipe_id: str, url: str):
    try:
        description = ""
        ydl_opts_meta = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 15,
        }
        with yt_dlp.YoutubeDL(ydl_opts_meta) as ydl:
            info = ydl.extract_info(url, download=False)
            description = info.get('description', '') or ''

        recipe_data = None
        if len(description.strip()) > 40:
            prompt_check = f"""
            Przeanalizuj opis z rolki kulinarnej:
            \"\"\"{description}\"\"\"

            Czy ten opis zawiera składniki ORAZ kroki przygotowania dania?
            Jeśli TAK, przetłumacz na polski i zwróć JSON:
            {{
                "has_recipe": true,
                "title": "Tytuł dania",
                "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie"}}],
                "instructions": ["krok 1", "krok 2"]
            }}
            Jeśli NIE, zwróć: {{"has_recipe": false}}
            """
            try:
                res = ask_gemini_with_retry(prompt_check)
                if res.get("has_recipe"):
                    recipe_data = res
            except Exception as e:
                print(f"Błąd analizy tekstu opisu: {e}")

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
                        print(f"Błąd audio: {audio_err}")

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

        if recipe_data and supabase:
            supabase.table("recipes").update({
                "title": recipe_data.get("title", "Przepis z Reels"),
                "ingredients": recipe_data.get("ingredients", []),
                "instructions": recipe_data.get("instructions", []),
                "status": "completed"
            }).eq("id", recipe_id).execute()

    except Exception as err:
        print(f"Błąd dla przepisu {recipe_id}: {err}")
        if supabase:
            supabase.table("recipes").update({"status": "error"}).eq("id", recipe_id).execute()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    recipes = []
    queue = []
    if supabase:
        try:
            res = supabase.table("recipes").select("*").order("created_at", desc=True).execute()
            for r in (res.data or []):
                if r.get("status") == "completed":
                    recipes.append(r)
                else:
                    queue.append(r)
        except Exception as e:
            print(f"Błąd bazy: {e}")
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"recipes": recipes, "queue": queue}
    )

@app.get("/api/status")
def get_status_api():
    """Zwraca liczbę zadań, by odświeżać widok po zakończeniu pobierania."""
    if supabase:
        res = supabase.table("recipes").select("id, status").execute()
        data = res.data or []
        completed_count = sum(1 for r in data if r.get("status") == "completed")
        queue_count = sum(1 for r in data if r.get("status") in ["pending", "error"])
        return {"completed": completed_count, "queue": queue_count}
    return {"completed": 0, "queue": 0}

@app.post("/process")
def process_reels(background_tasks: BackgroundTasks, url: str = Form(...)):
    url = url.strip()
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})
    
    res = supabase.table("recipes").insert({
        "title": "Pobieranie w toku...",
        "source_url": url,
        "status": "pending",
        "ingredients": [],
        "instructions": []
    }).execute()
    
    recipe_id = res.data[0]["id"]
    background_tasks.add_task(process_recipe_by_id, recipe_id, url)
    return {"status": "queued", "id": recipe_id}

@app.post("/retry/{recipe_id}")
def retry_recipe(recipe_id: str, background_tasks: BackgroundTasks):
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Błąd bazy"})

    res = supabase.table("recipes").select("*").eq("id", recipe_id).execute()
    if not res.data:
        return JSONResponse(status_code=404, content={"message": "Nie znaleziono"})
    
    recipe = res.data[0]
    supabase.table("recipes").update({
        "title": "Ponawianie próby...",
        "status": "pending"
    }).eq("id", recipe_id).execute()

    background_tasks.add_task(process_recipe_by_id, recipe_id, recipe["source_url"])
    return {"status": "retrying"}

@app.post("/delete/{recipe_id}")
def delete_recipe(recipe_id: str):
    """Usuwa wpis (z kolejki lub gotowy przepis) z bazy Supabase."""
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak połączenia z bazą"})
    try:
        supabase.table("recipes").delete().eq("id", recipe_id).execute()
        return {"status": "deleted", "id": recipe_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
