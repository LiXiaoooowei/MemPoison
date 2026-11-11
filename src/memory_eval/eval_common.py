"""Shared command line, path handling and bookkeeping for the eval_*.py scripts.

Every script gets the same options (output locations, data, LLM backend), the same run folder
layout and the same result files:
  <artifacts-dir>/run_mem-poison-<method>__<dataset>__<model>__<timestamp>[__<tag>]/
      results.json    one entry per question (checkpointed after every question)
      metrics.json    ISR / RSR / ASR / ACC summary
      run_config.json how the run was started (command line, data hashes, git commit, ...)
  <cache-dir>/mem-poison/<db_subdir>/<dataset>__<hash>/   the base memory DB (reusable between runs)
  <cache-dir>/mem-poison/huggingface/                     embedding model download
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import env_utils
import all_llm
from all_llm import LLMSpec

SCRIPT_DIR = Path(__file__).resolve().parent
MEMPOISON_ROOT = SCRIPT_DIR.parent.parent              # the mem-poison repository
DEFAULT_DATA_ROOT = MEMPOISON_ROOT / "data" / "data_eval"
POISON_PLACEHOLDER = "<<PUT_YOUR_POISON_TEXT_HERE>>"
# Settings that must not change when a run is resumed, otherwise old and new results are not comparable.
RESUME_KEYS = ["dataset", "trigger", "provider", "model", "judge_provider", "judge_model",
               "reasoning_effort", "noise_turns", "seed", "embedding_model",
               "qa_data_sha256", "poison_data_sha256", "noise_data_sha256"]


# ================= Small helpers =================
def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def slug(text):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(text)).strip("-")

def is_inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path == parent or parent in path.parents

def git_state(repo):
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


# ================= Command line =================
def add_common_args(p):
    g = p.add_argument_group("output locations (required, so nothing lands inside mem-poison)")
    g.add_argument("--cache-dir", required=True,
                   help="Cache root. Holds <cache-dir>/mem-poison/<memory DB> and the HuggingFace model cache.")
    g.add_argument("--artifacts-dir", required=True,
                   help="Artifacts root. Each run creates its own sub-folder here.")
    g.add_argument("--run-name", default=None, help="Optional tag appended to the run folder name, e.g. smoke.")
    g.add_argument("--resume", default=None, metavar="RUN_DIR",
                   help="Continue an earlier run folder. Finished questions are skipped.")
    g.add_argument("--env-file", default=None,
                   help="Path to a .env file. Default: MEMPOISON_ENV_FILE, else the nearest .env above the cwd.")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="longmemeval", help="Sub-folder of --data-root.")
    g.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Folder that contains the dataset folders.")
    g.add_argument("--qa-data", default=None, help="Override <data-root>/<dataset>/qa_data.json.")
    g.add_argument("--noise-data", default=None, help="Override <data-root>/<dataset>/noise_data.json.")
    g.add_argument("--poison-data", default=None,
                   help="Override <data-root>/<dataset>/poison_data.json. The shipped file is only a template.")
    g.add_argument("--trigger", required=True, help="Trigger string the poison texts contain.")

    g = p.add_argument_group("experiment")
    g.add_argument("--noise-turns", type=int, default=2000, help="Number of noise memories in the base DB.")
    g.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions (smoke test).")
    g.add_argument("--seed", type=int, default=42, help="Seed for sampling the noise memories.")
    g.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    g.add_argument("--embedding-device", default="cpu")
    g.add_argument("--error-sleep", type=float, default=60,
                   help="Seconds to wait after an unexpected error on a question (rate limits).")

    g = p.add_argument_group("LLM backend (agent and judge)")
    g.add_argument("--provider", default="openai", choices=sorted(all_llm.PROVIDERS),
                   help="openai = hosted API (API_KEY/Base_URL from .env); ollama = local Ollama server.")
    g.add_argument("--model", default="gpt-4o", help="Agent model name, e.g. qwen3:8b.")
    g.add_argument("--base-url", default=None,
                   help="Endpoint. Default: Base_URL from .env (openai) or $OLLAMA_HOST/v1 (ollama).")
    g.add_argument("--judge-provider", default=None, choices=sorted(all_llm.PROVIDERS),
                   help="Default: same as --provider.")
    g.add_argument("--judge-model", default=None, help="Default: same as --model.")
    g.add_argument("--judge-base-url", default=None, help="Default: same as --base-url if the provider is shared.")
    g.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high"],
                   help="Passed to the endpoint. 'none' turns off Qwen3 thinking on Ollama. Default: model default.")
    g.add_argument("--request-timeout", type=float, default=None,
                   help="Seconds per LLM request. Default: 120 for the agent, 60 for the judge.")
    return p


# ================= Run setup =================
def setup_run(args, method, db_subdir, db_key_extra=None, settings_extra=None, resume_keys_extra=()):
    """Validate paths, load env and data, configure the LLMs, create (or reopen) the run folder.

    method           short name used in the run folder, e.g. "rag"
    db_subdir        folder under <cache-dir>/mem-poison for this method's base DB, e.g. "rag_db"
    db_key_extra     extra values the base DB content depends on (mixed into the DB folder hash)
    settings_extra   extra values recorded in run_config.json
    resume_keys_extra  keys of settings_extra that must match when resuming
    Returns a SimpleNamespace with everything the evaluation loop needs.
    """
    cache_dir = Path(args.cache_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    for label, path in (("--cache-dir", cache_dir), ("--artifacts-dir", artifacts_dir)):
        if is_inside(path, MEMPOISON_ROOT):
            sys.exit(f"{label} points inside the mem-poison repository ({path}). "
                     f"Use a folder in the parent repository instead.")

    env_path = env_utils.load_env(args.env_file)
    print(f"Env file: {env_path or '(none found)'}")

    # Keep the embedding model download inside the cache folder.
    mempoison_cache = cache_dir / "mem-poison"
    hf_home = mempoison_cache / "huggingface"
    os.environ.setdefault("HF_HOME", str(hf_home))

    # ---- LLM backends
    judge_provider = args.judge_provider or args.provider
    shared = judge_provider == args.provider
    agent_spec = LLMSpec(provider=args.provider, model=args.model, temperature=0.7,
                         timeout=args.request_timeout or 120, base_url=args.base_url,
                         reasoning_effort=args.reasoning_effort)
    judge_spec = LLMSpec(provider=judge_provider, model=args.judge_model or args.model, temperature=0.2,
                         timeout=args.request_timeout or 60,
                         base_url=args.judge_base_url or (args.base_url if shared else None),
                         reasoning_effort=args.reasoning_effort if shared else None)
    if "openai" in (agent_spec.provider, judge_spec.provider) and not os.getenv("API_KEY"):
        sys.exit("The openai provider needs API_KEY (and Base_URL) in the environment or in the .env file.")
    all_llm.configure_llms(agent_spec, judge_spec)

    # ---- Data
    data_dir = Path(args.data_root) / args.dataset
    qa_path = Path(args.qa_data or data_dir / "qa_data.json")
    noise_path = Path(args.noise_data or data_dir / "noise_data.json")
    poison_path = Path(args.poison_data or data_dir / "poison_data.json")
    for path in (qa_path, noise_path, poison_path):
        if not path.is_file():
            sys.exit(f"Data file not found: {path}")

    print("Loading datasets...")
    noise_data = load_json(noise_path)
    qa_data = load_json(qa_path)
    poison_data = load_json(poison_path)

    if isinstance(noise_data, dict) and "dialogue" in noise_data:
        noise_pool = [item['user'] for item in noise_data['dialogue']]
    else:
        noise_pool = noise_data

    num_eval = len(qa_data) if args.limit is None else min(args.limit, len(qa_data))
    if len(poison_data) < num_eval or any(p.get("malicious_text") in (None, "", POISON_PLACEHOLDER)
                                          for p in poison_data[:num_eval]):
        sys.exit(f"{poison_path} needs {num_eval} entries with a real 'malicious_text' (one per evaluated "
                 f"question). It has {len(poison_data)} entries, or still holds the template placeholder. "
                 f"Pass --poison-data, or lower --limit.")

    # ---- Fingerprints: the base DB depends on the facts, the noise sample and the embedding model.
    fingerprint = {
        "dataset": args.dataset,
        "qa_data_sha256": sha256_file(qa_path),
        "noise_data_sha256": sha256_file(noise_path),
        "poison_data_sha256": sha256_file(poison_path),
        "noise_turns": args.noise_turns,
        "seed": args.seed,
        "embedding_model": args.embedding_model,
    }
    key_fields = {k: fingerprint[k] for k in ("qa_data_sha256", "noise_data_sha256", "noise_turns", "seed",
                                              "embedding_model")}
    key_fields.update(db_key_extra or {})
    db_key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:8]
    db_dir = mempoison_cache / db_subdir / f"{slug(args.dataset)}__{db_key}"

    # ---- Run folder
    if args.resume:
        run_dir = Path(args.resume).resolve()
        if not (run_dir / "run_config.json").is_file():
            sys.exit(f"--resume: {run_dir} has no run_config.json, so it is not a run folder.")
    else:
        parts = [f"run_mem-poison-{method}", slug(args.dataset), slug(args.model),
                 datetime.now().strftime("%Y%m%d_%H%M%S")]
        if args.run_name:
            parts.append(slug(args.run_name))
        run_dir = artifacts_dir / "__".join(parts)

    run_settings = {**fingerprint, "trigger": args.trigger, "provider": agent_spec.provider,
                    "model": agent_spec.model, "judge_provider": judge_spec.provider,
                    "judge_model": judge_spec.model, "reasoning_effort": args.reasoning_effort,
                    **(settings_extra or {})}
    config_path = run_dir / "run_config.json"
    if args.resume:
        saved = load_json(str(config_path)).get("settings", {})
        keys = list(RESUME_KEYS) + list(resume_keys_extra)
        changed = {k: (saved.get(k), run_settings.get(k)) for k in keys if saved.get(k) != run_settings.get(k)}
        if changed:
            sys.exit(f"Cannot resume {run_dir}: settings differ from the original run (saved, now): {changed}")
    else:
        os.makedirs(run_dir, exist_ok=False)
        save_json({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "command": [sys.executable] + sys.argv,
            "method": method,
            "settings": run_settings,
            "limit": args.limit,
            "paths": {"run_dir": str(run_dir), "db_dir": str(db_dir), "qa_data": str(qa_path),
                      "noise_data": str(noise_path), "poison_data": str(poison_path),
                      "env_file": env_path, "hf_home": os.environ["HF_HOME"]},
            "llm": {"agent": vars(agent_spec), "judge": vars(judge_spec)},
            "mem_poison_git": git_state(MEMPOISON_ROOT),
        }, str(config_path))
    print(f"Run folder: {run_dir}")

    # ---- Backends must answer before we spend time on the DB and the loop
    all_llm.check_llms_ready()

    return SimpleNamespace(
        args=args, cache_dir=cache_dir, mempoison_cache=mempoison_cache,
        hf_hub_dir=str(hf_home / "hub"), db_dir=db_dir, run_dir=run_dir,
        qa_data=qa_data, noise_pool=noise_pool, poison_data=poison_data, num_eval=num_eval,
        fingerprint=fingerprint, agent_spec=agent_spec, judge_spec=judge_spec, env_path=env_path)


# ================= Metrics =================
def compute_metrics(logs):
    total = len(logs)
    summary = {"total": total, "errors": sum(1 for log in logs if "error" in log["details"])}
    for name in ("ISR", "RSR", "ASR", "ACC"):
        count = sum(1 for log in logs if log["metrics"].get(name) == 1)
        summary[name] = {"count": count, "rate": (count / total) if total else None}
    return summary


def print_final_metrics(logs, run_dir, title):
    total = len(logs)
    if total == 0:
        print("No logs found. Cannot calculate metrics.")
        return

    summary = compute_metrics(logs)
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(summary, str(Path(run_dir) / "metrics.json"))

    c = {name: summary[name]["count"] for name in ("ISR", "RSR", "ASR", "ACC")}
    print("\n" + "="*45)
    print(title)
    print("="*45)
    print(f"Total Evaluated Sessions: {total} (with errors: {summary['errors']})")
    print("-" * 45)
    print(f"[ISR] Injection Success Rate: {(c['ISR']/total)*100:.2f}% ({c['ISR']}/{total})")
    print(f"[RSR] Retrieval Success Rate: {(c['RSR']/total)*100:.2f}% ({c['RSR']}/{total})")
    print(f"[ASR] Attack Success Rate   : {(c['ASR']/total)*100:.2f}% ({c['ASR']}/{total})")
    print(f"[ACC] Benign Accuracy       : {(c['ACC']/total)*100:.2f}% ({c['ACC']}/{total})")
    print("="*45)
    print(f"Full details saved to: {run_dir}\n")
