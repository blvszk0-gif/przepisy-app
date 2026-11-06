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
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    active_models = []
    try:
        models_data = groq_client.models.list().data
        for m in models_data:
            mid = m.id.lower()
            is_active = getattr(m, 'active', True)
            is_specialized = any(x in mid for x in ["whisper", "guard", "orpheus", "embed", "tts", "vision"])
            if is_active and not is_specialized and "/" not in m.id:
                active_models.append(m.id)
    except Exception as e:
        print(f"Błąd modeli Groq: {e}")

    if not active_models:
        active_models = ["llama-3.1-8b-instant"]

    active_models.sort(key=lambda name: 0 if "llama-3.1-8b" in name else (1 if "llama-3.3" in name else 2))

    system_instruction = (
        "Jesteś precyzyjnym asystentem kulinarnym. Twoim jedynym celem jest wyodrębnienie "
        "rzeczywistych składników i instrukcji z tekstu użytkownika. "
        "Zwracasz czysty JSON. Nigdy nie kopiujesz nazw pól ani słów kluczowych jako wartości."
    )

    for model_name in active_models:
        try:
            resp = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2048
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception:
            try:
                resp = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2048
                )
                match = re.search(r'\{.*\}', resp.choices[0].message.content.strip(), re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                continue

    raise Exception("Brak dostępnego modelu w Groq")

def ask_ai_smart(prompt: str):
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
                    time.sleep(3)
                    continue
                break
    return ask_groq_ai(prompt)

def sanitize_recipe_data(raw_data: dict, fallback_description: str = "") -> dict:
    """Weryfikuje, czy AI nie zwróciło zaślepek z szablonu, i oczyszcza dane."""
    raw_title = str(raw_data.get("title", "")).strip()
    
    # Blokada kopiowania zaślepek tytułu
    if not raw_title or "apetyczny tytuł" in raw_title.lower() or "przepis z reels" in raw_title.lower():
        # Pobieramy pierwszą sensowną linijkę z opisu jako tytuł
        lines = [l.strip() for l in fallback_description.split("\n") if len(l.strip()) > 3]
        raw_title = lines[0][:60] if lines else "Przepis z Instagrama"

    dummy_phrases = [
        "dokładna nazwa", "kategoria w markecie", "ilość produktu", 
        "przygotowanie...", "krok 1:", "krok 2:", "kombinuj składniki", "żywy produkt"
    ]

    cleaned_ingredients = []
    seen = set()
    banned_words = {"składniki", "składnik", "ingredients", "ingredient", "żywność", "krew", "śmieci"}

    for item in raw_data.get("ingredients", []):
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            cat = str(item.get("category", "Inne")).strip() or "Inne"
        else:
            name = str(item).strip()
            cat = "Inne"

        name_lower = name.lower()
        # Wykrycie i odrzucenie przekopiowanej zaślepki
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
        s = re.sub(r'\bZłość\b', 'Złóż / wymieszaj', s, flags=re.IGNORECASE)
        if s and len(s) > 3 and s_lower not in seen_steps:
            seen_steps.add(s_lower)
            cleaned_instructions.append(s)

    # Awaryjne wyciąganie z opisu posta, gdy model zawiedzie na całej linii
    if not cleaned_ingredients and fallback_description:
        desc_lines = [l.strip() for l in fallback_description.split("\n") if l.strip()]
        for l in desc_lines:
            if any(char in l for char in ["-", "•", "*", "g ", "ml", "łyż", "szt"]):
                cleaned_ingredients.append({"name": l.lstrip("-•* "), "category": "Produkty"})

    if not cleaned_instructions and fallback_description:
        desc_lines = [l.strip() for l in fallback_description.split("\n") if len(l.strip()) > 15]
        cleaned_instructions = desc_lines[:8]

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
            "progress_text": "Weryfikacja krzyżowa opisu i audio...",
            "transcript": transcript_text
        }).eq("id", recipe_id).execute()

        # BEZPIECZNY PROMPT BEZ TEKSTOWYCH ZAŚLEPEK DO SKOPIOWANIA
        prompt = f"""
        Przeanalizuj poniższe materiały kulinarne i wyciągnij z nich faktyczny przepis:

        TEKST Z OPISU POSTA:
        \"\"\"{description}\"\"\"

        TRANSKRYPCJA AUDIO:
        \"\"\"{transcript_text}\"\"\"

        ZASADY:
        1. Jeśli w tekście opisu znajdują się składniki i proporcje, wyciągnij je dokładnie.
        2. Odrzuć z audio wszelkie nielogiczne słowa, teksty piosenek lub szumy.
        3. Do każdego składnika dopasuj kategorię (np. Nabiał, Warzywa i owoce, Sypkie, Mięso, Przyprawy, Inne).
        4. Rozpisz kolejne czynności przygotowania na czytelne punkty.

        ZWRÓĆ WYŁĄCZNIE CZYSTY OBIEKT JSON W TYM FORMACIE:
        {{
            "title": "nazwa potrawy ustalona na podstawie tekstu",
            "ingredients": [
                {{"name": "nazwa produktu z gramaturą lub ilością", "category": "dział sklepu"}}
            ],
            "instructions": [
                "opis pierwszej czynności",
                "opis drugiej czynności"
            ]
        }}
        """

        raw_recipe = ask_ai_smart(prompt)
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
