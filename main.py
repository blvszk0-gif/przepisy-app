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

def extract_audio_and_transcribe(url: str):
    """Pomocnicza funkcja pobierająca sam dźwięk i robiąca transkrypcję w Groq."""
    transcript_text = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts_audio = {
            'format': 'ba/b',
            'outtmpl': f'{temp_dir}/audio.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 25,
            'max_filesize': 35 * 1024 * 1024,
        }
        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([url])

        downloaded_files = glob.glob(f"{temp_dir}/*")
        if downloaded_files and GROQ_API_KEY:
            media_file = downloaded_files[0]
            groq_client = Groq(api_key=GROQ_API_KEY)
            with open(media_file, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(os.path.basename(media_file), f.read()),
                    model="whisper-large-v3"
                )
            transcript_text = transcription.text
    return transcript_text

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
        has_transcript = False
        transcript_text = ""

        # KROK A: Szybka próba z samego opisu
        if len(description.strip()) > 40:
            prompt_check = f"""
            Przeanalizuj opis z rolki kulinarnej:
            \"\"\"{description}\"\"\"

            Czy ten opis zawiera dokładne składniki ORAZ kroki przygotowania dania?
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

        # KROK B: Jeśli brak pełnego przepisu w opisie -> pobieramy dźwięk
        if not recipe_data:
            try:
                transcript_text = extract_audio_and_transcribe(url)
                if transcript_text:
                    has_transcript = True
            except Exception as e:
                print(f"Transkrypcja nie powiodła się: {e}")

            prompt_full = f"""
            Stwórz przepis kulinarny na podstawie danych:
            Opis z posta: {description}
            Transkrypcja mowy z wideo: {transcript_text}

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
                "status": "completed",
                "has_transcript": has_transcript,
                "transcript": transcript_text
            }).eq("id", recipe_id).execute()

    except Exception as err:
        print(f"Błąd dla przepisu {recipe_id}: {err}")
        if supabase:
            supabase.table("recipes").update({"status": "error"}).eq("id", recipe_id).execute()

def enrich_recipe_with_audio_task(recipe_id: str, url: str):
    """Zadanie w tle wymuszające pobranie audio i uzupełnienie istniejącego przepisu."""
    try:
        # Pobieramy dotychczasowe dane z bazy
        current = supabase.table("recipes").select("*").eq("id", recipe_id).execute().data[0]
        
        # Pobieramy transkrypcję
        transcript_text = extract_audio_and_transcribe(url)
        
        # Prosimy Gemini o wzbogacenie przepisu o nowe informacje z mowy
        prompt_enrich = f"""
        Mamy istniejący przepis kulinarny:
        Tytuł: {current.get('title')}
        Składniki: {json.dumps(current.get('ingredients', []), ensure_ascii=False)}
        Instrukcje: {json.dumps(current.get('instructions', []), ensure_ascii=False)}

        Oto nowo uzyskana transkrypcja mowy twórcy z filmu:
        \"\"\"{transcript_text}\"\"\"

        Zadanie: Uzupełnij i uściślij listę składników oraz kroki przygotowania o wszelkie detale, triki, czasy pieczenia/smażenia i proporcje wspomniane w mowie.
        Przetłumacz na polski i zwróć zaktualizowany JSON:
        {{
            "title": "{current.get('title')}",
            "ingredients": [{{"name": "produkt i ilość", "category": "dział w sklepie"}}],
            "instructions": ["krok 1", "krok 2"]
        }}
        """
        enriched_data = ask_gemini_with_retry(prompt_enrich)

        supabase.table("recipes").update({
            "ingredients": enriched_data.get("ingredients", current.get("ingredients")),
            "instructions": enriched_data.get("instructions", current.get("instructions")),
            "has_transcript": True,
            "transcript": transcript_text,
            "status": "completed"
        }).eq("id", recipe_id).execute()
        print(f"Pomyślnie uzupełniono przepis {recipe_id} o transkrypcję audio!")

    except Exception as e:
        print(f"Błąd przy dociąganiu transkrypcji: {e}")
        supabase.table("recipes").update({"status": "completed"}).eq("id", recipe_id).execute()

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
    if supabase:
        res = supabase.table("recipes").select("id, status, has_transcript").execute()
        data = res.data or []
        completed_count = sum(1 for r in data if r.get("status") == "completed")
        queue_count = sum(1 for r in data if r.get("status") in ["pending", "error"])
        transcripts_count = sum(1 for r in data if r.get("has_transcript"))
        return {"completed": completed_count, "queue": queue_count, "transcripts": transcripts_count}
    return {"completed": 0, "queue": 0, "transcripts": 0}

@app.post("/process")
def process_reels(background_tasks: BackgroundTasks, url: str = Form(...)):
    url = url.strip()
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})
    
    res = supabase.table("recipes").insert({
        "title": "Pobieranie w toku...",
        "source_url": url,
        "status": "pending",
        "has_transcript": False,
        "ingredients": [],
        "instructions": []
    }).execute()
    
    recipe_id = res.data[0]["id"]
    background_tasks.add_task(process_recipe_by_id, recipe_id, url)
    return {"status": "queued", "id": recipe_id}

@app.post("/enrich-audio/{recipe_id}")
def enrich_audio(recipe_id: str, background_tasks: BackgroundTasks):
    """Endpoint wymuszający dociągnięcie transkrypcji z audio."""
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak bazy"})
    
    res = supabase.table("recipes").select("*").eq("id", recipe_id).execute()
    if not res.data:
        return JSONResponse(status_code=404, content={"message": "Nie znaleziono"})
    
    recipe = res.data[0]
    background_tasks.add_task(enrich_recipe_with_audio_task, recipe_id, recipe["source_url"])
    return {"status": "started"}

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
    if not supabase:
        return JSONResponse(status_code=500, content={"message": "Brak połączenia z bazą"})
    try:
        supabase.table("recipes").delete().eq("id", recipe_id).execute()
        return {"status": "deleted", "id": recipe_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
