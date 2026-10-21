"""Wire dialects. Each module parses and validates its own request shape and builds its own replies; the shared
splitters, accumulator and template layer live one level up."""
from slotstream_spark.server.dialects import anthropic, ollama, openai_chat, responses

DIALECTS = {"openai_chat": openai_chat, "ollama": ollama, "anthropic": anthropic, "responses": responses}
