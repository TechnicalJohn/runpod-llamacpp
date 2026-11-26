"""
Module engine

This module provides lightweight adapters that present an OpenAI-compatible
interface for local llama.cpp. It contains two primary classes:

- LlamaCPPEngine: A front-facing adapter that accepts a JobInput and transforms
  it into OpenAI-compatible routes and payloads. It normalizes prompt vs chat
  input and delegates actual interaction to an OpenAI-style engine.
- LlamaCPPOpenAIEngine: A concrete engine that uses the OpenAI client (pointing
  at a locally hosted inference service) to list models and create completions
  or chat completions. It supports both non-streaming and streaming responses.

Environment:
- The OpenAI client is configured to point at a local base_url in this script.

Typical usage:
- Construct LlamaCPPEngine() and call its async generate() with a JobInput.
- The generator yields either dicts (non-stream responses) or streaming chunks
  formatted as strings when stream=True.
"""

import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:3098/v1/",
    api_key="none",
)


class LlamaCPPEngine:
    """
    Adapter that prepares JobInput for an OpenAI-style engine and yields
    responses.

    This class normalizes incoming JobInput.llm_input into an OpenAI-style
    payload. If llm_input is a string, it prepares a completions request; if
    it's a list (messages), it prepares a chat.completions request. It then
    delegates the actual generation to a lower-level OpenAI-compatible engine
    (LlamaCPPOpenAIEngine).

    Attributes:
        model_id (str): Optional pre-resolved model ID.
    """

    def __init__(self, model_id: str = None):
        """
        Initialize the LlamaCPPEngine.
        """
        self.model_id = model_id
        print("Llama.cpp engine initialized")

    async def generate(self, job_input):
        """
        Asynchronously generate responses for a given JobInput.
        """
        openAIEngine = LlamaCPPOpenAIEngine(model_id=self.model_id)

        # Resolve model ID once if not provided
        model = self.model_id
        if not model:
            models_resp = client.models.list()
            model = models_resp.data[0].id

        if isinstance(job_input.llm_input, str):
            openAIjob = JobInput(
                {
                    "openai_route": "/v1/completions",
                    "openai_input": {
                        "model": model,
                        "prompt": job_input.llm_input,
                        "stream": job_input.stream,
                    },
                }
            )
        else:
            openAIjob = JobInput(
                {
                    "openai_route": "/v1/chat/completions",
                    "openai_input": {
                        "model": model,
                        "messages": job_input.llm_input,
                        "stream": job_input.stream,
                    },
                }
            )

        async for batch in openAIEngine.generate(openAIjob):
            yield batch


class LlamaCPPOpenAIEngine(LlamaCPPEngine):
    """
    Concrete OpenAI-compatible engine that uses the OpenAI client to perform
    model listing and completions against a locally hosted inference service.
    """

    def __init__(self, model_id: str = None):
        super().__init__(model_id)
        print("LlamaCPPOpenAIEngine initialized")

    async def generate(self, job_input):
        """
        Dispatch the provided OpenAI-style JobInput to the appropriate handler.
        """
        openai_input = job_input.openai_input

        if job_input.openai_route == "/v1/models":
            async for response in self._handle_model_request():
                yield response
        elif job_input.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(
                openai_input,
                chat=job_input.openai_route == "/v1/chat/completions",
            ):
                yield response
        else:
            raise ValueError(f"Invalid route: {job_input.openai_route}")

    async def _handle_model_request(self):
        """
        Handle a model-listing request using the OpenAI client.
        """
        try:
            response = client.models.list()
            yield {
                "object": "list",
                "data": [model.to_dict() for model in response.data],
            }
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise e

    async def _handle_chat_or_completion_request(self, openai_input, chat=False):
        """
        Handle a chat or completion request and yield responses or streaming
        chunks.
        """
        try:
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            if not openai_input.get("stream", False):
                yield response.to_dict()
                return

            for chunk in response:
                yield "data: " + json.dumps(
                    chunk.to_dict(), separators=(",", ":")
                ) + "\n\n"

            yield "data: [DONE]"
        except Exception as e:
            logger.error(f"Inference request failed: {str(e)}")
            raise e
