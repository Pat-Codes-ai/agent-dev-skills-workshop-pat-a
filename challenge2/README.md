# Challenge 2: Enhancing the Weather Chatbot with Callbacks

The Challenge 1 chatbot plus three ADK callbacks. Nothing else changed.

Notebook: `challenge2_weather_callbacks.ipynb` (Colab Enterprise).

| Requirement | Implementation |
|---|---|
| Callback to log user prompts | `log_user_prompt`, a `before_model_callback` that writes `USER » ...` to the log |
| Callback to log model responses | `log_model_response`, an `after_model_callback` that writes `MODEL » ...` (text or tool call) |
| Validate input before it reaches the model | `moderate_user_prompt`, a `before_model_callback` that asks a small Gemini model (`check_user_input`) for a one-word verdict |
| 3a. Location must be in the U.S. | Verdict `NON_US` returns a canned refusal `LlmResponse`, so the model and tools never run |
| 3b. Input must not be malicious | Verdict `BAD` returns a canned refusal the same way |

`chained_before_callback` runs moderation first and logging second, and is what the agent is wired
to. Returning `None` from a callback lets the model call continue; returning an `LlmResponse` stops
it and sends that response back to the user.

## Running it

Same as Challenge 1: upload to Colab Enterprise (region `us-central1`), connect to the default
runtime, run top to bottom. Section 2b tests the classifier on six prompts, Section 5b proves the
callback blocks a Canadian city and a prompt-injection attempt while a normal question still works.

Section 7 writes the agent as the `weather_bot_v2/` package so you can chat with it from the
notebook's Terminal tab with `adk run weather_bot_v2` and watch the log lines live.
