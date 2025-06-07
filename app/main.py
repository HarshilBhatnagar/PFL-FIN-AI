from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.routes.chat import router as chat_router
from app.routes.ingest import router as ingest_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Financial Document Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(chat_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")

@app.get("/")
async def root(request: Request):
    logger.info("Root endpoint accessed")
    return templates.TemplateResponse("index.html", {"request": request})
