#!/usr/bin/env python3
"""Judge evaluator: use an LLM to re-score saved evaluation traces.

Usage:
  python scripts/judge_evaluator.py [path/to/results.json]

Behavior:
 - Loads an evaluation results JSON (or picks the latest in evaluation/results/)
 - For each successful eval, builds a judgment prompt containing query,
   system answer, extracted metadata, filters, and reference answer (if available)
 - Calls the configured LLM (Azure OpenAI or Ollama) and expects a JSON
   response with numeric scores (0.0-1.0) and short rationale
 - If Langfuse is configured and a trace_id exists in the eval result, the
   script logs the judge scores into Langfuse as numeric scores
 - Saves an augmented results file with judge outputs appended

This script supports the project's configuration in `src.config`.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src import config
from evaluation.evaluation_dataset import get_evaluation_by_id, get_evaluation_dataset


JUDGE_PROMPT = """
You are an impartial evaluator. Given the following information about a RAG query,
produce a JSON object with the following numeric scores (0.0 - 1.0) and a
short rationale for each:

- metadata_extraction: correctness of extracted metadata vs expected
- filter_usage: whether applied filters match expected intent
- retrieval_relevance: whether retrieved documents (or none) are relevant
- answer_fidelity: whether the answer strictly uses only provided documents
- citation_accuracy: whether cited filenames actually support the answer
- overall: weighted overall score (0.0-1.0)

Return strictly valid JSON like:
{
  "metadata_extraction": 0.8,
  "filter_usage": 0.7,
  "retrieval_relevance": 0.6,
  "answer_fidelity": 0.9,
  "citation_accuracy": 0.9,
  "overall": 0.78,
  "rationales": {
    "metadata_extraction": "short explanation",
    ...
  }
}

Input:
Query: {query}
Reference answer: {reference}
System answer: {answer}
Extracted metadata: {extracted}
Expected metadata: {expected}
Applied filters: {filters}
Retrieved documents (filenames or preview): {docs}

Judge fairly and be conservative when unsure.
"""


def call_azure_openai(prompt: str) -> str:
    endpoint = config.AZURE_OPENAI_ENDPOINT or config.LLM_URL
    if not endpoint or not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Azure OpenAI not configured in config")

    deployment = config.AZURE_OPENAI_DEPLOYMENT or config.LLM_MODEL
    api_version = config.AZURE_OPENAI_API_VERSION or "2024-06-01"
    url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    body = {
        "messages": [
            {"role": "system", "content": "You are a concise evaluation assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
    }

    headers = {
        "api-key": config.AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    # Azure response -> choices[0].message.content
    return out["choices"][0]["message"]["content"]


def call_ollama(prompt: str) -> str:
    url = (config.OLLAMA_URL or config.LLM_URL)
    if not url:
        raise RuntimeError("Ollama URL not configured")
    model = config.OLLAMA_MODEL or config.LLM_MODEL
    api_url = url.rstrip("/") + "/chat"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
    resp = requests.post(api_url, json=body, timeout=60)
    resp.raise_for_status()
    out = resp.json()
    # Ollama returns {"id":..., "choices":[{"message":{"content": ...}}]}
    return out.get("choices", [])[0].get("message", {}).get("content", "")


def call_llm(prompt: str) -> str:
    llm_type = getattr(config, "LLM_TYPE", None) or config.MODEL_TO_USE
    if llm_type == "AZURE_OPENAI":
        return call_azure_openai(prompt)
    elif llm_type == "OLLAMA":
        return call_ollama(prompt)
    else:
        raise RuntimeError(f"Unsupported LLM type: {llm_type}")


def extract_answer_text(answer_preview: str) -> str:
    # Try to be robust: many saved answers are ChatMessage(...) reprs —
    # extract quoted text if present, otherwise return raw preview.
    import re
    m = re.search(r"TextContent\(text=\"(.*)\"\)", answer_preview)
    if m:
        return m.group(1)
    # fallback: remove ChatMessage wrapper
    s = answer_preview
    s = s.replace("ChatMessage", "").strip()
    return s


def main(results_path: Optional[Path] = None):
    results_dir = Path("evaluation/results")
    if results_path is None:
        # pick latest file
        files = sorted(results_dir.glob("evaluation_results_*.json"), reverse=True)
        if not files:
            print("No evaluation result files found in evaluation/results/")
            return
        results_path = files[0]

    print(f"Loading results from: {results_path}")
    data = json.loads(results_path.read_text(encoding="utf-8"))

    augmented = data.copy()
    augmented["judge_runs"] = []

    # optional Langfuse
    lf = None
    try:
        if config.LANGFUSE_ENABLED:
            from langfuse import Langfuse
            lf = Langfuse()
            print("Langfuse client initialized — judge will log scores when trace_id present")
    except Exception:
        lf = None

    dataset = {item["id"]: item for item in get_evaluation_dataset()}

    for r in data.get("results", []):
        if not r.get("success"):
            continue

        eval_id = r.get("eval_id")
        ds_item = dataset.get(eval_id)
        reference = ds_item.get("reference_answer") if ds_item else ""
        docs = r.get("answer_preview", "")
        prompt = JUDGE_PROMPT.format(
            query=r.get("query", ""),
            reference=reference,
            answer=extract_answer_text(r.get("answer_preview", "")),
            extracted=json.dumps(r.get("metadata", {}).get("extracted", {}), ensure_ascii=False),
            expected=json.dumps(r.get("metadata", {}).get("expected", {}), ensure_ascii=False),
            filters=json.dumps(r.get("filters", {}), ensure_ascii=False),
            docs=docs,
        )

        try:
            llm_out = call_llm(prompt)
        except Exception as e:
            print(f"LLM call failed for {eval_id}: {e}")
            continue

        # Try to parse JSON from LLM output
        judge_json = None
        try:
            judge_json = json.loads(llm_out)
        except Exception:
            # Attempt to extract JSON substring
            import re
            m = re.search(r"\{[\s\S]*\}", llm_out)
            if m:
                try:
                    judge_json = json.loads(m.group(0))
                except Exception as e:
                    print(f"Failed to parse JSON for {eval_id}: {e}")

        if not judge_json:
            print(f"No valid JSON returned for {eval_id}. Raw output:\n{llm_out}\n")
            continue

        augmented["judge_runs"].append({"eval_id": eval_id, "judge": judge_json})

        # Log into Langfuse if available and trace_id present
        trace_id = r.get("_internal", {}).get("langfuse_trace_id") or r.get("trace_id")
        if lf and trace_id:
            try:
                for name, value in judge_json.items():
                    if name == "rationales":
                        continue
                    lf.create_score(
                        name=f"judge_{name}",
                        value=float(value),
                        trace_id=trace_id,
                        comment=f"Judge score for {eval_id}",
                        data_type="NUMERIC",
                    )
            except Exception as e:
                print(f"Failed to send judge scores to Langfuse for {eval_id}: {e}")

    out_path = Path("evaluation/results") / (results_path.stem + "_judged.json")
    out_path.write_text(json.dumps(augmented, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Augmented results with judge outputs saved to: {out_path}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(p)
