from typing import Optional, Any, Dict, Mapping
import pandas as pd
import os, json, time, yaml, re, inspect, random
from functools import wraps
from datetime import datetime

# Heavy ML imports are deferred to classes that need them (model_pipeline, model_regular).
# This allows API-based detectors to work without torch/transformers installed.
torch = None
transformers = None
F = None
softmax = None

def _ensure_torch():
    """Lazy-load torch and transformers on first use."""
    global torch, transformers, F, softmax
    if torch is None:
        import torch as _torch
        torch = _torch
        torch._dynamo.config.cache_size_limit = 64
        import torch.nn.functional as _F
        F = _F
        from torch.nn.functional import softmax as _softmax
        softmax = _softmax
        import transformers as _transformers
        transformers = _transformers

import botocore
import openai
import boto3
import requests
from openai import OpenAI


try:
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
except ImportError:
    discovery = None
    HttpError = Exception

try:
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.ai.contentsafety.models import AnalyzeTextOptions
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    ContentSafetyClient = None
    AnalyzeTextOptions = None
    AzureKeyCredential = None

from dotenv import load_dotenv
load_dotenv()

perspective_api_key = os.getenv("perspective_api_key")
gemini_api_key = os.getenv("gemini_api_key")
openai_api_key = os.getenv("openai_api_key")
enkryptai_api_key = os.getenv("enkryptai_api_key")
azure_api_key = os.getenv("azure_api_key")
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")


BYTE_LIMIT = 1000
SPLIT_REGEX = re.compile(r'(?<=[.!?。！？])\s+')

def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))

def _clean(s: str) -> str:
    return (s or "").replace("\u200b","").replace("\ufeff","").strip()

def _safe_truncate(s: str, limit: int = BYTE_LIMIT) -> str:
    s = _clean(s)
    if _byte_len(s) <= limit:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if _byte_len(s[:mid]) <= limit:
            lo = mid + 1
        else:
            hi = mid
    return s[:lo-1].rstrip()

def _chunk_by_bytes(text: str, limit: int = BYTE_LIMIT):
    t = _clean(text)
    if _byte_len(t) <= limit:
        return [t]
    sentences = [s for s in SPLIT_REGEX.split(t) if s]
    chunks, cur = [], ""
    def fits(c): return _byte_len(c) <= limit

    for s in sentences:
        if _byte_len(s) > limit:
            buf = ""
            for w in s.split():
                cand = (buf + " " + w).strip()
                if fits(cand):
                    buf = cand
                else:
                    if buf: chunks.append(buf)
                    if _byte_len(w) > limit:
                        chunks.append(_safe_truncate(w, limit))
                        buf = ""
                    else:
                        buf = w
            if buf: chunks.append(buf)
        else:
            cand = (cur + " " + s).strip() if cur else s
            if fits(cand):
                cur = cand
            else:
                chunks.append(cur)
                cur = s
    if cur: chunks.append(cur)
    return chunks


class model_pipeline:
    def __init__(self, model_name: str, download_path: str):
        _ensure_torch()
        self.model_name = model_name
        self.model_id = os.path.join(download_path, model_name)
        self.pipeline = transformers.pipeline(
            task="text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.float16},
            tokenizer=self.model_id,
            device=0
        )
        self.tokenizer = self.pipeline.tokenizer
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def detect(self,
        text: str,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> str:
        """Return raw generated text from the pipeline without any post-processing."""
        output = self.pipeline(
            text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False,
            truncation=True
        )[0]["generated_text"]

        return output.strip()


class model_regular:
    def __init__(self, model_name: str, download_path: str, classification_models: list[str]):
        _ensure_torch()
        self.model_name = model_name
        self.classification_models = set(classification_models)
        self.model_id = os.path.join(download_path, model_name)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        model_cls = (
            transformers.AutoModelForCausalLM
            if model_name not in self.classification_models or "gemma" in self.model_name or "phi-2" in self.model_name
            else transformers.AutoModelForSequenceClassification
        )
        self.model = model_cls.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )

        if self.classification_models:
            self.config = transformers.AutoConfig.from_pretrained(self.model_id)

    def detect(
        self,
        text: str,
        tau: float = 1.0,
        bias_yes: float = 0.0,
        bias_no: float = 0.0,
        min_margin: float = 0.0
    ):
        """Return raw model outputs without verdict/threshold post-processing."""
        if self.model_name in self.classification_models:
            if "gemma" in self.model_name or "phi-2" in self.model_name:
                inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                last = logits[:, -1, :]
                
                tid_yes = (self.tokenizer.encode(" Yes", add_special_tokens=False) or self.tokenizer.encode("Yes", add_special_tokens=False))[0]
                tid_no  = (self.tokenizer.encode(" No",  add_special_tokens=False) or self.tokenizer.encode("No",  add_special_tokens=False))[0]
                
                sel = torch.stack([last[0, tid_yes], last[0, tid_no]])

                if tau <= 0:
                    raise ValueError("tau must be > 0")
                sel = sel / tau

                if bias_yes != 0.0 or bias_no != 0.0:
                    sel = sel + torch.tensor([bias_yes, bias_no], device=sel.device, dtype=sel.dtype)
                
                probs = softmax(sel, dim=0)
                p_yes, p_no = float(probs[0]), float(probs[1])

                return {
                    "UNSAFE_score": p_yes,
                    "SAFE_score": p_no,
                    "tau": tau,
                    "bias_yes": bias_yes,
                    "bias_no": bias_no,
                    "min_margin": min_margin,
                }
            else:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt", 
                    truncation=True
                ).to("cuda")
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits.float()
                C = logits.shape[-1]
                id2label = getattr(self.config, "id2label", None) or {}

                if C > 2:
                    probs = torch.sigmoid(logits[0])
                    raw = [{"label": id2label.get(i, f"LABEL_{i}"), "score": float(p)}
                        for i, p in enumerate(probs)]
                    return {"raw": raw}

                elif C == 1:
                    p = float(torch.sigmoid(logits[0, 0]))
                    lab = id2label.get(0, "toxic")
                    return {"raw": [{"label": lab, "score": p}]}

                else:
                    probs = F.softmax(logits[0], dim=-1)
                    raw = [{"label": id2label.get(i, f"LABEL_{i}"), "score": float(p)} for i, p in enumerate(probs)]
                    return {"raw": raw}
            
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            text_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id 
                )
            generated = outputs[0][text_len:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            
            return response


class model_api:
    def __init__(self, model_name: str, max_retries: int = 5):
        self.model_name = model_name
        self.max_retries = max_retries

        name = model_name.lower()
        self.client = None
        self.api_url = None
        self.headers = None
        self.params = None
        self.detectors = None
        self.url = None
        self.comprehend = None
        self.bedrock = None

        if "gemini" in name:
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            self.headers = {"Content-Type": "application/json"}
            self.params = {"key": gemini_api_key}

        elif model_name == "perspective_api":
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=perspective_api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

        elif model_name == "azure_ai_content_safety_api":
            self.client = ContentSafetyClient(
                "https://kate-try-2.cognitiveservices.azure.com",
                AzureKeyCredential(azure_api_key)
            )

        elif model_name == "comprehend":
            self.comprehend = boto3.client(
                service_name="comprehend",
                region_name="us-east-1",
                aws_access_key_id=client_id,
                aws_secret_access_key=client_secret
            )

        elif "titan" in name:
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id=client_id,
                aws_secret_access_key=client_secret
            )

        elif name == "enkryptai":
            self.detectors = {
                "nsfw": {"enabled": True},
                "toxicity": {"enabled": True},
                "injection_attack": {"enabled": True},
                "policy_violation": {
                    "enabled": True,
                    "need_explanation": True,
                    "policy_text": "Do not allow any illegal or immoral activities."
                }
            }
            self.headers = {"Content-Type": "application/json", "apikey": enkryptai_api_key}
            self.url = "https://api.enkryptai.com/guardrails/detect"

        elif model_name == "omni-moderation-latest" or "gpt" in name:
            self.client = OpenAI(api_key=openai_api_key)

    def detect(
        self,
        text: str,
        temperature: float = 0,
        top_p: float = 0.8,
        top_k: int = 10,
    ):
        time.sleep(1)
        
        name = self.model_name.lower()

        if "gemini" in name:
            config = {"temperature": temperature, "topK": top_k, "topP": top_p, "candidateCount": 1}
            payload = {
                "systemInstruction": {"parts": [{"text": "You are a compliance classifier. Answer Yes or No only."}]},
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": config
            }
            retries = 0
            while retries < self.max_retries:
                try:
                    r = requests.post(self.api_url, headers=self.headers, params=self.params, json=payload)
                    r.raise_for_status()
                    result = r.json()
                    result["generationConfig"] = config
                    return result
                
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in 10 seconds...")
                    time.sleep(10)
                    retries += 1
            return "Error: Maximum retries reached"

        if name == "perspective_api":
            payload = {
                "comment": {"text": text},
                "requestedAttributes": {"TOXICITY": {}},
                "languages": ["en"],
                "doNotStore": True,
            }

            response = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.comments().analyze(body=payload).execute()
                    return response
                except HttpError as e:
                    status = getattr(e.resp, "status", None)
                    ra = None
                    try:
                        ra = e.resp.get("Retry-After")
                    except Exception:
                        pass
                    sleep_s = float(ra) if ra else min(30, (1.6 ** attempt) * (1 + random.random()))
                    print(f"HTTP {status} -> retry {attempt+1}/{self.max_retries}, sleep {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue
                except Exception as e:
                    sleep_s = min(30, (1.6 ** attempt) * (1 + random.random()))
                    print(f"Unexpected {e} -> retry {attempt+1}/{self.max_retries}, sleep {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue

            return {"error": "ERROR_RETRY_EXCEEDED"}
        
        if name == "azure_ai_content_safety_api":
            request = AnalyzeTextOptions(
                text=text,
                categories=["Hate", "SelfHarm", "Sexual", "Violence"]
            )
            response = self.client.analyze_text(request)
            return {
                "categoriesAnalysis": [
                    {"category": item.category, "severity": item.severity}
                    for item in response.categories_analysis
                ]
            }

        if "gpt" in name or name == "omni-moderation-latest":
            retries = 0
            while retries < self.max_retries:
                try:
                    if name == "omni-moderation-latest":
                        resp = self.client.moderations.create(
                            input=text, model="omni-moderation-latest"
                        )
                        return resp.model_dump()
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name, messages=[{"role": "user", "content": text}]
                        )
                        return resp.model_dump()
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in 10 seconds...")
                    time.sleep(10); retries += 1
            return "Error: Maximum retries reached"

        if name == "comprehend":
            return self._detect_comprehend_raw(text)

        if "titan" in name:
            body = json.dumps({
                "inputText": text,
                "textGenerationConfig": {"temperature": temperature, "maxTokenCount": 5, "topP": top_p}
            })
            response = self.bedrock.invoke_model(
                modelId=self.model_name,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            return json.loads(response["body"].read())

        if name == "enkryptai":
            payload = {"text": text, "detectors": self.detectors}
            retries = 0
            while retries < 5:
                try:
                    r = requests.post(self.url, headers=self.headers, json=payload)
                    return r.json()
                except Exception as e:
                    print("Error parsing response:", e)
                    try:
                        print("Raw response text:", r.text)
                    except Exception:
                        pass
                    time.sleep(10); retries += 1
            return "Error: Maximum retries reached"
    
    def _detect_comprehend_raw(self, text: str) -> dict:
        chunks = _chunk_by_bytes(text, BYTE_LIMIT)

        raw_segments = []
        for ch in chunks:
            try:
                resp = self.comprehend.detect_toxic_content(
                    TextSegments=[{"Text": ch}],
                    LanguageCode="en",
                )
            except botocore.exceptions.ClientError as e:
                raw_segments.append({"error": str(e), "chunk": ch})
                continue
            raw_segments.append(resp)

        return {
            "chunks": len(chunks),
            "raw_segments": raw_segments,
        }


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f">>> timer starts <<< start running: {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f">>> timer ends <<< {func.__name__} is finisheed, duration is {duration:.4f} seconds.")
        return result
    return wrapper