import json
import os
import random
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from .json_utils import extract_json_obj


RETRYABLE_STATUS_CODES = {408, 409, 429}
DEFAULT_USER_AGENT = "skill-evolution/1.0"
CONTEXT_RETRY_MAX_SAFETY_TOKENS = 256
CONTEXT_RETRY_MIN_SAFETY_TOKENS = 8


class NonRetryableOpenAIError(RuntimeError):
    pass


def _retry_label(attempt: int, max_retries: int) -> str:
    if int(max_retries) < 0:
        return f"{attempt + 1}/infinite"
    return f"{attempt + 1}/{max_retries}"


def _should_stop_retrying(attempt: int, max_retries: int) -> bool:
    return int(max_retries) >= 0 and attempt >= int(max_retries)


def _retry_sleep_seconds(initial_sleep: float, max_sleep: float, attempt: int) -> float:
    sleep_s = min(float(max_sleep), float(initial_sleep) * (2 ** min(int(attempt), 30)))
    return sleep_s * (0.8 + 0.4 * random.random())


def _is_retryable_http_error(status_code: int, response_text: str) -> bool:
    status_code = int(status_code)
    if status_code in RETRYABLE_STATUS_CODES or 500 <= status_code < 600:
        return True
    lower = str(response_text or "").lower()
    return (
        "upstream_error" in lower
        or "upstream request failed" in lower
        or "bad gateway" in lower
    )


def _context_length_retry_budget(
    status_code: int,
    response_text: str,
    current_max_tokens: int,
) -> Optional[Dict[str, int]]:
    if int(status_code) != 400:
        return None
    text = str(response_text or "")
    if "context length" not in text.lower():
        return None

    context_match = re.search(
        r"maximum context length is\s*([\d,]+)\s*tokens",
        text,
        flags=re.I,
    )
    input_match = re.search(
        r"prompt contains(?: at least)?\s*([\d,]+)\s*input tokens",
        text,
        flags=re.I,
    )
    if input_match is None:
        input_match = re.search(
            r"parameter\s*=\s*input_tokens\s*,\s*value\s*=\s*([\d,]+)",
            text,
            flags=re.I,
        )
    if context_match is None or input_match is None:
        return None

    context_limit = int(context_match.group(1).replace(",", ""))
    input_tokens = int(input_match.group(1).replace(",", ""))
    current_budget = max(1, int(current_max_tokens))
    available = context_limit - input_tokens
    input_is_lower_bound = bool(
        re.search(r"prompt contains\s+at least\s*[\d,]+\s*input tokens", text, flags=re.I)
    )
    safety_tokens = min(
        CONTEXT_RETRY_MAX_SAFETY_TOKENS,
        max(CONTEXT_RETRY_MIN_SAFETY_TOKENS, available // 16),
    )
    adjusted_budget = min(current_budget - 1, available - safety_tokens)
    if input_is_lower_bound:
        adjusted_budget = min(adjusted_budget, current_budget // 2)
    if adjusted_budget < 1:
        return None
    return {
        "context_limit": context_limit,
        "input_tokens": input_tokens,
        "max_tokens": adjusted_budget,
        "safety_tokens": safety_tokens,
    }


def _generic_context_retry_budget(
    status_code: int,
    response_text: str,
    current_max_tokens: int,
    context_attempt: int,
) -> Optional[Dict[str, int]]:
    """Back off completion length when a proxy omits token counts in 400s."""

    if int(status_code) != 400:
        return None
    text = str(response_text or "").lower()
    if not any(
        phrase in text
        for phrase in ("context window", "context length", "maximum context")
    ):
        return None
    # Keep this bounded: if the prompt itself is too large, repeated retries
    # cannot help and the caller should receive the original non-retryable error.
    if int(context_attempt) >= 5:
        return None
    current = max(1, int(current_max_tokens))
    adjusted = max(128, current // 2)
    if adjusted >= current:
        return None
    return {
        "context_limit": 0,
        "input_tokens": 0,
        "max_tokens": adjusted,
        "safety_tokens": 0,
    }


def _default_reasoning_effort(model: str, reasoning_effort: Optional[str]) -> Optional[str]:
    if reasoning_effort:
        return reasoning_effort
    name = str(model or "").lower()
    if name.startswith("gpt-5") and "mini" in name:
        return "medium"
    return None


def _summarize_error_text(text: str, limit: int = 180) -> str:
    text = str(text or "")
    title = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    if title:
        summary = title.group(1)
    else:
        summary = re.sub(r"<[^>]+>", " ", text)
    summary = re.sub(r"\s+", " ", summary).strip()
    if len(summary) > limit:
        summary = summary[:limit].rstrip() + "..."
    return summary or "no response body"


def _extract_choice_text(data: Dict[str, Any]) -> str:
    try:
        choice = (data.get("choices") or [])[0]
    except Exception as exc:
        raise RuntimeError(f"OpenAI response has no choices: {str(data)[:800]}") from exc
    if not isinstance(choice, dict):
        raise RuntimeError(f"OpenAI response choice is not an object: {str(choice)[:800]}")

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    chunks.append(str(part.get("text") or part.get("content") or ""))
                else:
                    chunks.append(str(part))
            text = "".join(chunks).strip()
            if text:
                return text
        for key in ("reasoning_content", "reasoning", "refusal"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value

    text = choice.get("text")
    if isinstance(text, str):
        return text

    raise RuntimeError(
        "OpenAI response choice does not contain message.content or text. "
        f"Response prefix: {json.dumps(data, ensure_ascii=False)[:1200]}"
    )


class OpenAIChatClient:
    def __init__(
        self,
        model: str,
        api_key: str = None,
        base_url: str = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        timeout: int = 180,
        max_retries: int = -1,
        retry_initial_sleep: float = 5.0,
        retry_max_sleep: float = 60.0,
        debug_dir: str = None,
        thinking: str = None,
        reasoning_effort: str = None,
        extra_body: Optional[Dict[str, Any]] = None,
        context_length_retry: bool = False,
        credential_env_prefix: str = None,
        proxy_url: str = None,
    ):
        import os
        import requests

        self.requests = requests
        self.model = model

        env_prefix = str(credential_env_prefix or "").strip().upper()
        scoped_api_key_name = f"{env_prefix}_OPENAI_API_KEY" if env_prefix else None
        scoped_base_url_name = f"{env_prefix}_OPENAI_BASE_URL" if env_prefix else None
        self.api_key = (
            api_key
            or (os.environ.get(scoped_api_key_name) if scoped_api_key_name else None)
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or (os.environ.get(scoped_base_url_name) if scoped_base_url_name else None)
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        self.credential_env_prefix = env_prefix or None
        scoped_proxy_name = f"{env_prefix}_OPENAI_PROXY_URL" if env_prefix else None
        self.proxy_url = (
            proxy_url
            or (os.environ.get(scoped_proxy_name) if scoped_proxy_name else None)
            or os.environ.get("OPENAI_PROXY_URL")
            or ""
        ).rstrip("/")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_initial_sleep = float(retry_initial_sleep)
        self.retry_max_sleep = float(retry_max_sleep)
        self.debug_dir = debug_dir or os.environ.get("OPENAI_DEBUG_DIR")
        self.thinking = thinking
        self.reasoning_effort = _default_reasoning_effort(model, reasoning_effort)
        self.extra_body = dict(extra_body or {})
        self.context_length_retry = bool(context_length_retry)
        self.user_agent = (
            os.environ.get("WMSM_USER_AGENT")
            or os.environ.get("OPENAI_USER_AGENT")
            or DEFAULT_USER_AGENT
        )

        if not self.api_key:
            expected = (
                f"{scoped_api_key_name} or OPENAI_API_KEY"
                if scoped_api_key_name
                else "OPENAI_API_KEY"
            )
            raise RuntimeError(f"OpenAI-compatible API key is not set ({expected})")

        if not self.base_url:
            expected = (
                f"{scoped_base_url_name} or OPENAI_BASE_URL"
                if scoped_base_url_name
                else "OPENAI_BASE_URL"
            )
            raise RuntimeError(f"OpenAI-compatible base URL is not set ({expected})")

    def _dump_debug(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        response: Any = None,
        status_code: int = None,
        error: Any = None,
    ) -> None:
        if not self.debug_dir:
            return
        try:
            path = os.path.join(self.debug_dir, "openai_chat_debug.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            messages = payload.get("messages") or []
            row = {
                "id": uuid.uuid4().hex,
                "time": time.time(),
                "kind": kind,
                "model": self.model,
                "base_url": self.base_url,
                "user_agent": self.user_agent,
                "status_code": status_code,
                "payload_keys": sorted(payload.keys()),
                "message_count": len(messages) if isinstance(messages, list) else None,
                "message_chars": [
                    len(str((m or {}).get("content", ""))) if isinstance(m, dict) else len(str(m))
                    for m in messages
                ] if isinstance(messages, list) else None,
                "payload": payload,
                "response": response,
                "error": str(error) if error is not None else None,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            return

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        resolved_temperature = self.temperature if temperature is None else temperature
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        payload.update(self.extra_body)
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        last_error = None
        attempt = 0
        context_attempt = 0
        while True:
            try:
                request_kwargs = {
                    "headers": headers,
                    "json": payload,
                    "timeout": self.timeout,
                }
                if self.proxy_url:
                    request_kwargs["proxies"] = {
                        "http": self.proxy_url,
                        "https": self.proxy_url,
                    }
                resp = self.requests.post(url, **request_kwargs)
                if resp.status_code < 400:
                    data = resp.json()
                    try:
                        text = _extract_choice_text(data)
                        if text.strip():
                            return text
                        raise RuntimeError(
                            "OpenAI response choice content is empty. "
                            f"Response prefix: {json.dumps(data, ensure_ascii=False)[:1200]}"
                        )
                    except RuntimeError as exc:
                        self._dump_debug(
                            kind="empty_or_unparseable_choice",
                            payload=payload,
                            response=data,
                            status_code=resp.status_code,
                            error=exc,
                        )
                        last_error = exc
                        if _should_stop_retrying(attempt, self.max_retries):
                            break
                        sleep_s = _retry_sleep_seconds(self.retry_initial_sleep, self.retry_max_sleep, attempt)
                        print(
                            f"[llm retry] model={self.model} attempt={_retry_label(attempt, self.max_retries)} "
                            f"sleep={sleep_s:.1f}s error={last_error}",
                            flush=True,
                        )
                        time.sleep(sleep_s)
                        attempt += 1
                        continue

                last_error = RuntimeError(
                    f"OpenAI request failed: status={resp.status_code} "
                    f"body={_summarize_error_text(resp.text)}"
                )
                self._dump_debug(
                    kind="http_error",
                    payload=payload,
                    response=resp.text,
                    status_code=resp.status_code,
                    error=last_error,
                )
                if not _is_retryable_http_error(resp.status_code, resp.text):
                    context_retry = (
                        _context_length_retry_budget(
                            resp.status_code,
                            resp.text,
                            int(payload.get("max_tokens") or 1),
                        )
                    if self.context_length_retry
                        else None
                    )
                    if context_retry is None and self.context_length_retry:
                        context_retry = _generic_context_retry_budget(
                            resp.status_code,
                            resp.text,
                            int(payload.get("max_tokens") or 1),
                            context_attempt,
                        )
                    if context_retry is not None:
                        previous_budget = int(payload.get("max_tokens") or 1)
                        payload["max_tokens"] = context_retry["max_tokens"]
                        print(
                            f"[llm context retry] model={self.model} "
                            f"attempt={context_attempt + 1} "
                            f"input_tokens={context_retry['input_tokens']} "
                            f"context_limit={context_retry['context_limit']} "
                            f"safety_tokens={context_retry['safety_tokens']} "
                            f"max_tokens={previous_budget}->{context_retry['max_tokens']}",
                            flush=True,
                        )
                        context_attempt += 1
                        continue
                    raise NonRetryableOpenAIError(str(last_error))
            except self.requests.exceptions.RequestException as exc:
                self._dump_debug(
                    kind="request_exception",
                    payload=payload,
                    error=exc,
                )
                last_error = exc

            if _should_stop_retrying(attempt, self.max_retries):
                break

            sleep_s = _retry_sleep_seconds(self.retry_initial_sleep, self.retry_max_sleep, attempt)
            print(
                f"[llm retry] model={self.model} attempt={_retry_label(attempt, self.max_retries)} "
                f"sleep={sleep_s:.1f}s error={last_error}",
                flush=True,
            )
            time.sleep(sleep_s)
            attempt += 1

        raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

    def chat_json(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        last_error = None
        attempt = 0
        while True:
            text = ""
            try:
                text = self.chat(messages)
                obj = extract_json_obj(text)
                if obj is not None:
                    return obj
                last_error = ValueError(
                    "Model did not return valid JSON. "
                    f"Raw output prefix:\n{text[:1000]}"
                )
            except NonRetryableOpenAIError:
                raise
            except Exception as exc:
                last_error = exc

            if _should_stop_retrying(attempt, self.max_retries):
                break

            sleep_s = _retry_sleep_seconds(self.retry_initial_sleep, self.retry_max_sleep, attempt)
            print(
                f"[llm json retry] model={self.model} attempt={_retry_label(attempt, self.max_retries)} "
                f"sleep={sleep_s:.1f}s error={last_error}",
                flush=True,
            )
            time.sleep(sleep_s)
            attempt += 1

        raise RuntimeError(f"OpenAI JSON request failed after retries: {last_error}")

class HFChatClient:
    """Small local HF generation client compatible with chat_json.

    This supports local Hugging Face inference. For large models, prefer serving the model with vLLM
    and using OpenAIChatClient against the local OpenAI-compatible endpoint.
    """
    def __init__(self, model_name_or_path: str, temperature: Optional[float] = None, max_tokens: int = 1024,
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = str(model_name_or_path)
        adapter_config = os.path.join(model_path, "adapter_config.json")
        base_model_name = model_path
        self.adapter_path = None
        if os.path.exists(adapter_config):
            try:
                with open(adapter_config, "r", encoding="utf-8") as f:
                    adapter_meta = json.load(f)
                base_model_name = adapter_meta.get("base_model_name_or_path") or model_path
                self.adapter_path = model_path
            except Exception:
                base_model_name = model_path
        kwargs = {'trust_remote_code': True, 'device_map': 'auto'}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs['torch_dtype'] = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        temp = self.temperature if temperature is None else temperature
        gen_kwargs = {
            **enc,
            "do_sample": bool(temp and temp > 0),
            "max_new_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(temp)
            gen_kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        return self.tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def chat_json(self, system_prompt: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
        ]
        text = self.chat(messages)
        obj = extract_json_obj(text)
        if obj is None:
            raise ValueError(f'Model did not return valid JSON: {text[:500]}')
        return obj


def build_llm_client(kind: str, model: str, api_key=None, base_url=None, temperature=None,
                     max_tokens=1024, load_in_4bit=False, timeout=180, max_retries=-1,
                     retry_initial_sleep=5.0, retry_max_sleep=60.0, debug_dir=None,
                     thinking=None, reasoning_effort=None, extra_body=None,
                     context_length_retry=False, credential_env_prefix=None,
                     proxy_url=None):
    if kind == 'hf':
        return HFChatClient(model, temperature=temperature, max_tokens=max_tokens, load_in_4bit=load_in_4bit)
    return OpenAIChatClient(
        model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_initial_sleep=retry_initial_sleep,
        retry_max_sleep=retry_max_sleep,
        debug_dir=debug_dir,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
        context_length_retry=context_length_retry,
        credential_env_prefix=credential_env_prefix,
        proxy_url=proxy_url,
    )
