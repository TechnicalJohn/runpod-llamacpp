"""
Runpod handler for processing jobs using LlamaCPP or OpenAI engines. This
module defines an asynchronous handler function that receives job inputs,
instantiates the appropriate engine based on the job input, and yields
generated output in a streaming fashion.
"""

import asyncio
import logging
import os
from typing import Any

import runpod
from engine import LlamaCPPEngine, LlamaCPPOpenAIEngine
from utils import JobInput

# set max concurrency from environment variable or default
DEFAULT_MAX_CONCURRENCY = 8

logger = logging.getLogger(__name__)
max_concurrency = int(os.getenv("MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY))


async def handler(job: Any):
    """
    Asynchronous handler function for processing jobs. It receives a job
    dictionary, extracts the input, determines the appropriate engine to use
    (LlamaCPP or OpenAI), and yields generated output in a streaming manner.
    """
    try:
        job_input = JobInput(job["input"])
        engine_class = (
            LlamaCPPOpenAIEngine if job_input.openai_route else LlamaCPPEngine
        )
        engine = engine_class()

        async for batch in engine.generate(job_input):
            if isinstance(batch, dict) and "error" in batch:
                raise Exception(batch["error"])
            yield batch

    except asyncio.CancelledError:
        logger.warning(f"Job {job.get('id')} was cancelled.")
        raise
    except Exception as e:
        logger.error(f"Job {job.get('id')} failed: {str(e)}", exc_info=True)
        yield {"error": str(e), "type": type(e).__name__}


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _x: max_concurrency,
        "return_aggregate_stream": True,
    }
)
