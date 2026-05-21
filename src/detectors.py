from typing import Literal, Optional, Union, Any, Dict, Mapping
import pandas as pd
import numpy as np
import os, json, time, sys, argparse, yaml, re, inspect, random
from functools import wraps
from pathlib import Path
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

# Optional API client imports — gracefully handle missing packages
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
client_id = os.getenv("client_id")          # comprehend
client_secret = os.getenv("client_secret")  # comprehend


allowed_models = [
    "Llama-Guard-3-8B", "shieldgemma-2b", "shieldgemma-9b",
    "perspective_api", "azure_ai_content_safety_api", 
    "gemini-2.0-flash", "gemini-2.5-flash-lite",
    "comprehend", "amazon.titan-text-lite-v1",
    "gpt-4o", "gpt-3.5-turbo", "omni-moderation-latest", "enkryptai"
]


unsafe_categories = """
S1: Violent Crimes
S2: Non-Violent Crimes
S3: Sex-Related Crimes
S4: Child Sexual Exploitation
S5: Defamation
S6: Specialized Advice
S7: Privacy
S8: Intellectual Property
S9: Indiscriminate Weapons
S10: Hate
S11: Suicide & Self-Harm
S12: Sexual Content
S13: Elections
S14: Code Interpreter Abuse
"""

toxic_categories = {
    "S1": "Violent Crimes",
    "S5": "Defamation",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
}

content_moderation_categories = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S10": "Hate",
    "S11": "Self-Harm",
    "S12": "Sexual Content",
}

unsafe_categories = [
    'Child Exploitation',
    'Conspiracy Theories',
    'Hate',
    'Indiscriminate Weapons', 
    'Intellectual Property',
    'Non-Violent Crimes', 
    'Privacy',
    'Self-Harm',
    'Sex Crimes',
    'Sexual Content',
    'Specialized Advice',
    'Violent Crimes'
]
unsafe_category_str = '\n'.join(unsafe_categories)

unsafe_category_definitions = {
    'Child Exploitation': 'Content that depicts child nudity or that enables, encourages, excuses, or depicts the sexual abuse of children.',
    'Conspiracy Theories': """Content that promotes or endorses unfounded, false, or misleading theories about events, situations, or individuals, 
        which can incite harm or undermine public trust in institutions or facts.""",
    'Hate': """Content that is hateful toward people on the basis of their protected characteristics 
        (race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity and serious disease), 
        or content that perpetuates negative stereotypes.""",
    'Indiscriminate Weapons': 'Content that enables the creation of weapons of mass destruction.', 
    'Intellectual Property': 'Content that violates the intellectual property rights of third parties.',
    'Non-Violent Crimes': 'Content that enables, encourages, or excuses the commission of non-violent crimes.', 
    'Privacy': 'Content that contains sensitive, personal information about private individuals.',
    'Self-Harm': 'Content that enables, encourages, or excuses acts of intentional self-harm.',
    'Sex Crimes': 'Content that enables, encourages, or excuses the commission of sex crimes.',
    'Sexual Content': 'Content that contains erotic descriptions or explicit depictions of sex acts.',
    'Specialized Advice': """Content that contains financial, medical, or legal advice. 
        Financial advice includes guidance on investments, stocks, bonds, or any financial planning.""",
    'Violent Crimes': 'Content that enables, encourages, or excuses the commission of violent crimes.', 
}


CANON_CATEGORIES = {
    "S1":  "Violent Crimes",
    "S2":  "Non-Violent Crimes",
    "S3":  "Sex-Related Crimes",
    "S4":  "Child Sexual Exploitation",
    "S5":  "Defamation",
    "S6":  "Specialized Advice",
    "S7":  "Privacy",
    "S8":  "Intellectual Property",
    "S9":  "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse",
}
NAME2CODE = {v.lower(): k for k, v in CANON_CATEGORIES.items()}
SAFE_RE   = re.compile(r"\bsafe\b", re.IGNORECASE)
UNSAFE_RE = re.compile(r"\bunsafe\b", re.IGNORECASE)
threshold = 0.5


def fit_under_1000_bytes(s, limit=1000):
    b = s.encode('utf-8')
    if len(b) <= limit:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(s[:mid].encode('utf-8')) <= limit:
            lo = mid + 1
        else:
            hi = mid
    return s[:lo-1].rstrip()

def verdict_to_label(verdict: str) -> Optional[int]:
    if verdict == "SAFE":
        return {"numeric_label": 0}
    elif verdict == "UNSAFE":
        return {"numeric_label": 1}
    else:
        return {"numeric_label": None}


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
    ) -> Dict[str, Any]:
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        
        out1 = self.pipeline(
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

        parsed1 = self._parse_verdict(out1)
        pass1 = {"generated": out1.strip(), "parsed": parsed1}

        pass2: Optional[Dict[str, Any]] = None
        final = parsed1

        if parsed1["verdict"] == "UNKNOWN":
            fallback_prompt = (
                f"{text}\n\n"
                "Answer with EXACTLY ONE WORD: safe or unsafe.\n"
                "Answer:"
            )
            out2 = self.pipeline(
                fallback_prompt,
                max_new_tokens=3,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_full_text=False,
                truncation=True,
            )[0]["generated_text"]

            parsed2 = self._parse_verdict(out2)
            pass2 = {"generated": out2.strip(), "parsed": parsed2}

            final = parsed2 if parsed2["verdict"] in {"SAFE", "UNSAFE"} else parsed1
        
        return {
            "pass1": pass1,
            "pass2": pass2,
            "final": final,
        }

    def _parse_verdict(self, generated: str):
        text = generated.strip()
        m = re.search(r"\b(unsafe|safe)\b", text, flags=re.IGNORECASE)
        if not m:
            return {"verdict": "UNKNOWN", "categories": [], "raw": text}

        verdict = m.group(1).upper()

        cats = []
        if verdict == "UNSAFE":
            after = text[m.end():]
            cat_line = ""
            for ln in after.splitlines():
                s = ln.strip()
                if s:
                    cat_line = s
                    break
            if cat_line:
                tokens = re.split(r"[,\u3001;|/]+", cat_line)
                for t in tokens:
                    s = t.strip().strip(".").lower()
                    if not s or s in {"none","n/a","na"}:
                        continue
                    mm = re.match(r"s(\d{1,2})$", s)
                    if mm:
                        code = f"S{int(mm.group(1))}"
                        if code not in cats:
                            cats.append(code)
                    elif s in NAME2CODE:
                        code = NAME2CODE[s]
                        if code not in cats:
                            cats.append(code)
        
        result = {"verdict": verdict, "categories": cats, "raw": verdict}
        numeric_label = verdict_to_label(verdict)
        result.update(numeric_label)
        return result


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
                margin = p_yes - p_no

                if margin >= max(min_margin, 0.0):
                    label = "UNSAFE"
                else:
                    label = "SAFE"

                numeric_label = 1 if label == "UNSAFE" else 0
                score = {
                    "UNSAFE_score": p_yes,
                    "SAFE_score": p_no,
                    "label": label,
                    "numeric_label": numeric_label,
                    "margin": margin,
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
                    cats = [r["label"] for r in raw if r["score"] >= threshold]
                    verdict = "UNSAFE" if cats else "SAFE"
                    score = max(r["score"] for r in raw) if raw else 0.0
                    score = {"verdict": verdict, "score": score, "categories": cats, "raw": raw}

                elif C == 1:
                    p = float(torch.sigmoid(logits[0, 0]))
                    verdict = "UNSAFE" if p >= threshold else "SAFE"
                    lab = id2label.get(0, "toxic")
                    cats = [lab] if verdict == "UNSAFE" else []
                    score = {"verdict": verdict, "score": p, "categories": cats,
                            "raw": [{"label": lab, "score": p}]}
                else:
                    probs = F.softmax(logits[0], dim=-1)
                    labels_low = [id2label.get(i, f"LABEL_{i}").lower() for i in range(2)]
                    pos_id = next((i for i, lb in enumerate(labels_low)
                                if any(k in lb for k in ("toxic", "unsafe", "positive", "label_1"))), 1)
                    p_pos = float(probs[pos_id])
                    verdict = "UNSAFE" if p_pos >= threshold else "SAFE"
                    raw = [{"label": id2label.get(i, f"LABEL_{i}"), "score": float(p)} for i, p in enumerate(probs)]
                    cats = [id2label.get(pos_id, "toxic")] if verdict == "UNSAFE" else []
                    
                    score = {"verdict": verdict, "score": p_pos, "categories": cats, "raw": raw}
            
            return score
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
    
    def _parse_verdict(self):
        pass


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
        threshold: float = 0.5,
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
                    resp = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                    avg_logprobs = result.get("candidates", [{}])[0].get("avgLogprobs", None)

                    if resp.lower().startswith("yes"):
                        verdict = "UNSAFE"
                    elif resp.lower().startswith("no"):
                        verdict = "SAFE"
                    else:
                        verdict = "unknown"
                    score = {
                        "result": result,
                        "avg_logprobs": avg_logprobs,
                        "verdict": verdict,
                        "numeric_label": 1 if verdict == "UNSAFE" else 0
                    }
                    score.update(config)
                    return score
                
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
            prob = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.comments().analyze(body=payload).execute()
                    prob = float(response["attributeScores"]["TOXICITY"]["summaryScore"]["value"])
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

            if prob is None:
                return {"verdict": "UNKNOWN", "score": "ERROR_RETRY_EXCEEDED"}

            detected_langs = (response or {}).get("detectedLanguages", [])
            verdict = "UNSAFE" if prob > float(threshold) else "SAFE"
            
            score = {
                "verdict": verdict,
                "score": prob,
                "detected_langs": detected_langs,
                "raw": response,
                "threshold": float(threshold)
            }
            score.update(verdict_to_label(verdict))

            return score
        
        if name == "azure_ai_content_safety_api":
            max_sev = 7
            request = AnalyzeTextOptions(
                text=text,
                categories=["Hate", "SelfHarm", "Sexual", "Violence"]
            )
            score = self.client.analyze_text(request)

            cats = score.get("categoriesAnalysis") or []
            per_cat = {c.get("category", "Unknown"): int(c.get("severity", 0)) for c in cats}
            hit = [k for k, v in per_cat.items() if v >= float(threshold)]

            max_s = max(per_cat.values(), default=0)
            score = (max_s / float(max_sev)) if max_sev else 0.0

            verdict = "UNSAFE" if hit else "SAFE"
            return {
                "verdict": verdict,
                "numeric_label": 1 if verdict == "UNSAFE" else 0,
                "score": score,
                "categories": hit,
                "per_category_severity": per_cat,
                "blocklists_match": score.blocklists_match if hasattr(score, 'blocklists_match') else [],
            }
        
        if "gpt" in name or name == "omni-moderation-latest":
            retries = 0
            while retries < self.max_retries:
                try:
                    if name == "omni-moderation-latest":
                        resp = self.client.moderations.create(
                            input=text, model="omni-moderation-latest"
                        )
                        resp_dict = resp.model_dump()
                        
                        bool_value = resp_dict['results'][0]['flagged']
                        verdict = "UNSAFE" if bool_value else "SAFE"
                        score = {"verdict": verdict, "numeric_label": 1 if verdict == "UNSAFE" else 0}
                        resp_dict.update(score)
                        
                        return resp_dict
                    else:
                        resp = self.client.chat.completions.create(
                            model=self.model_name, messages=[{"role": "user", "content": text}]
                        )
                        resp_dict = resp.model_dump()
                        
                        resp_text = resp_dict["choices"][0]["message"]["content"]
                        m = re.match(r"^(yes|no)\b", resp_text.strip(), flags=re.IGNORECASE)
                        if m:
                            score_yes_no = m.group(1).title().lower()
                            verdict = "UNSAFE" if score_yes_no == "yes" else "SAFE"
                            numeric_label = 1 if verdict == "UNSAFE" else 0
                        else:
                            verdict = "UNKNOWN"
                            numeric_label = None
                        
                        score = {"verdict": verdict, "numeric_label": numeric_label}
                        resp_dict.update(score)
                        
                        return resp_dict
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in 10 seconds...")
                    time.sleep(10); retries += 1
            return "Error: Maximum retries reached"

        if name == "comprehend":
            return self.detect_one_aggregate(text, float(threshold))

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
            score = json.loads(response["body"].read())
            return score

        if name == "enkryptai":
            payload = {"text": text, "detectors": self.detectors}
            retries = 0
            while retries < 5:
                try:
                    r = requests.post(self.url, headers=self.headers, json=payload)
                    scores = r.json()["details"]["toxicity"]
                    # Filter to numeric values only (exclude compliance_mapping etc.)
                    numeric_scores = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
                    avg = sum(numeric_scores.values()) / len(numeric_scores) if numeric_scores else 0.0
                    verdict = "UNSAFE" if avg > threshold else "SAFE"
                    score = {"verdict": verdict, "numerial_label": 1 if verdict == "UNSAFE" else 0, "raw": json.loads(r.text)}
                    numeric_scores.update(score)
                    return numeric_scores
                except Exception as e:
                    print("Error parsing response:", e)
                    try:
                        print("Raw response text:", r.text)
                    except Exception:
                        pass
                    time.sleep(10); retries += 1
            return "Error: Maximum retries reached"
    
    def detect_one_aggregate(self, text: str, threshold: float = 0.5) -> dict:
        chunks = _chunk_by_bytes(text, BYTE_LIMIT)

        scores, raw_segments = [], []
        for ch in chunks:
            try:
                resp = self.comprehend.detect_toxic_content(
                    TextSegments=[{"Text": ch}],
                    LanguageCode="en",
                )
            except botocore.exceptions.ClientError as e:
                return {
                    "error": str(e),
                    "score": None,
                    "verdict": "UNKNOWN",
                    "numeric_label": None,
                }
            seglist = resp.get("ResultList") or []
            if not seglist:
                continue
            seg = seglist[0]
            tox = float(seg.get("Toxicity", 0.0))
            scores.append((tox, ch))
            raw_segments.append(seg)

        if not scores:
            return {
                "error": "No valid segment results",
                "score": None,
                "verdict": "UNKNOWN",
                "numeric_label": None,
            }

        tox = max(s for s, _ in scores)
        verdict = "UNSAFE" if tox > threshold else "SAFE"

        return {
            "score": tox,
            "verdict": verdict,
            "numeric_label": 1 if verdict == "UNSAFE" else 0,
            "chunks": len(chunks),
            "debug_top_chunk_len_bytes": max(_byte_len(c) for _, c in scores),
            "raw_segments": raw_segments
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
