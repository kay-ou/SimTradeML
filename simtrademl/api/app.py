"""FastAPI application for model inference API.

Provides REST API endpoints for model predictions and management.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from simtrademl.api.routers import management, predict
from simtrademl.utils.logging import get_logger, trace_id_var

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        Control to the application
    """
    # Startup
    logger.info(
        "Starting SimTradeML Inference API",
        environment=settings.environment.value,
        api_port=settings.api_port,
    )

    yield

    # Shutdown
    logger.info("Shutting down SimTradeML Inference API")


# Create FastAPI application
app = FastAPI(
    title="SimTradeML Inference API",
    description="Machine Learning model inference API for financial trading",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development() else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(management.router)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Logging middleware with trace ID support.

    Args:
        request: FastAPI request
        call_next: Next middleware/handler

    Returns:
        Response from next handler
    """
    # Generate trace ID
    trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)

    # Log request
    start_time = time.time()
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        trace_id=trace_id,
    )

    # Process request
    try:
        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            trace_id=trace_id,
        )

        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = trace_id

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration_ms=round(duration * 1000, 2),
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "trace_id": trace_id,
            },
            headers={"X-Trace-ID": trace_id},
        )


@app.get("/")
async def root():
    """Root endpoint.

    Returns:
        Welcome message
    """
    return {
        "service": "SimTradeML Inference API",
        "version": "1.0.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "simtrademl.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development(),
        log_level=settings.log_level.value.lower(),
    )
