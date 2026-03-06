from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse

from app.routes import predict
from app.services.ml_model import KYCResponsePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="PRECOG",
    description="Predict. Prevent. Comply",
    version="1.0.0",
    docs_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/", tags=["Default"])
async def root() -> dict[str, object]:
    return {
        "message": "PRECOG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/v1/health",
            "/api/v1/predict/single",
            "/api/v1/predict/batch",
            "/api/v1/analytics/dashboard-data",
            "/api/v1/analytics/segment/{segment_name}",
        ],
    }


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
    ).body.decode("utf-8")

    custom_styles = """
    <style>
      body { background-color: #f5f1e8 !important; }
      .swagger-ui, .swagger-ui .wrapper, .swagger-ui .scheme-container { background-color: #f5f1e8 !important; }
      .swagger-ui .topbar { background-color: #f5f1e8 !important; border-bottom: 1px solid #ded8ca; }
      .swagger-ui .info .title {
        color: #000000 !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
      }
      .swagger-ui .info p {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
      }
      .swagger-ui .opblock-tag {
        text-transform: none !important;
      }
    </style>
    """
    html = html.replace("</head>", f"{custom_styles}</head>")
    return HTMLResponse(content=html)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("API starting...")
    model = KYCResponsePredictor(model_path="data/models/kyc_model.joblib")
    model.load_model("data/models/kyc_model.joblib")

    if model.model is not None:
        logger.info("API starting... Model loaded successfully")
    else:
        logger.warning(
            "API starting... Model file not found. API will start without loaded model."
        )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("API shutting down")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path,
        },
    )
