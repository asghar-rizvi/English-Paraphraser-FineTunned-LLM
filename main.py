from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from llm_handler import LLM_Handler

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize LLM Handler
handler = LLM_Handler('model/final_paraphraser')


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/paraphrase")
async def paraphrase_text(text: str = Form(...)):
    try:
        paraphrases = handler.paraphrase(text, num_return_sequences=3)
        return {
            "status": "success",
            "data": {
                "paraphrased_texts": paraphrases,
                "count": len(paraphrases)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }