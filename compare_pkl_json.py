#!/usr/bin/env python3
"""Compare a pickle file and a JSON file by structure and sampled values.

Usage:
  python3 compare_pkl_json.py --pkl encoded_data.pkl --json encoded_data.json
"""

import argparse
import json
import math
import pickle
import sys
from typing import Any, Dict, List


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:
        return None


NP = _try_import_numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PKL and JSON content")
    parser.add_argument("--pkl", required=True, help="Path to input pickle file")
    parser.add_argument("--json", required=True, help="Path to input json file")
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Max sampled items per list/dict level (default: 5)",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=100,
        help="Max mismatch entries to print (default: 100)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Float compare tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--reader-check",
        action="store_true",
        help="Also check whether data matches reader.py encoded_data top-level shape",
    )
    return parser.parse_args()


def load_pickle(path: str) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        msg = (
            f"Failed to load pickle: {e}.\\n"
            "This usually means the pickle contains numpy objects. "
            "Install numpy in the current env and retry."
        )
        raise RuntimeError(msg) from e


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_scalar(x: Any) -> Any:
    if NP is not None:
        if isinstance(x, NP.generic):
            return x.item()
    return x


def _is_number(x: Any) -> bool:
    x = _normalize_scalar(x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_sequence(x: Any) -> bool:
    if isinstance(x, (list, tuple)):
        return True
    if NP is not None and isinstance(x, NP.ndarray):
        return True
    return False


def _seq_len(x: Any) -> int:
    if NP is not None and isinstance(x, NP.ndarray):
        return int(x.shape[0]) if x.ndim > 0 else 1
    return len(x)


def _seq_get(x: Any, idx: int) -> Any:
    if NP is not None and isinstance(x, NP.ndarray):
        return x[idx]
    return x[idx]


def _type_name(x: Any) -> str:
    if NP is not None and isinstance(x, NP.ndarray):
        return f"ndarray(shape={x.shape}, dtype={x.dtype})"
    return type(x).__name__


class Comparator:
    def __init__(self, max_items: int, max_diffs: int, tol: float):
        self.max_items = max_items
        self.max_diffs = max_diffs
        self.tol = tol
        self.diffs: List[str] = []
        self.visited = 0

    def add_diff(self, msg: str) -> None:
        if len(self.diffs) < self.max_diffs:
            self.diffs.append(msg)

    def compare(self, a: Any, b: Any, path: str = "$") -> None:
        if len(self.diffs) >= self.max_diffs:
            return

        self.visited += 1
        a = _normalize_scalar(a)
        b = _normalize_scalar(b)

        if isinstance(a, dict) and isinstance(b, dict):
            self._compare_dict(a, b, path)
            return

        if _is_sequence(a) and _is_sequence(b):
            self._compare_seq(a, b, path)
            return

        if _is_number(a) and _is_number(b):
            aa = float(a)
            bb = float(b)
            if not math.isclose(aa, bb, rel_tol=self.tol, abs_tol=self.tol):
                self.add_diff(f"{path}: value mismatch {aa} != {bb}")
            return

        if isinstance(a, str) and isinstance(b, str):
            if a != b:
                self.add_diff(f"{path}: string mismatch {a!r} != {b!r}")
            return

        if a != b:
            self.add_diff(
                f"{path}: type/value mismatch ({_type_name(a)}={a!r}) != ({_type_name(b)}={b!r})"
            )

    def _compare_dict(self, a: Dict[Any, Any], b: Dict[Any, Any], path: str) -> None:
        # JSON turns non-string dict keys into strings; compare on string view.
        a_map = {str(k): k for k in a.keys()}
        b_map = {str(k): k for k in b.keys()}

        a_keys = set(a_map.keys())
        b_keys = set(b_map.keys())

        missing_in_json = sorted(a_keys - b_keys)
        missing_in_pkl = sorted(b_keys - a_keys)

        if missing_in_json:
            self.add_diff(f"{path}: keys missing in json (sample): {missing_in_json[:self.max_items]}")
        if missing_in_pkl:
            self.add_diff(f"{path}: keys missing in pkl (sample): {missing_in_pkl[:self.max_items]}")

        common = sorted(a_keys & b_keys)
        for key in common[: self.max_items]:
            self.compare(a[a_map[key]], b[b_map[key]], f"{path}.{key}")

    def _compare_seq(self, a: Any, b: Any, path: str) -> None:
        la = _seq_len(a)
        lb = _seq_len(b)
        if la != lb:
            self.add_diff(f"{path}: length mismatch {la} != {lb}")

        n = min(la, lb, self.max_items)
        for i in range(n):
            self.compare(_seq_get(a, i), _seq_get(b, i), f"{path}[{i}]")


def check_reader_encoded_shape(obj: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(obj, dict):
        return ["Top-level must be dict. Expected split dict: {'train','dev','test'} or turn_num dict."]

    # encoded_data.pkl can be either full split dict or single split turn dict.
    top_keys = set(str(k) for k in obj.keys())
    has_split = any(k in top_keys for k in ("train", "dev", "test"))

    to_check: List[tuple[str, Any]] = []
    if has_split:
        for split in ("train", "dev", "test"):
            if split in obj:
                to_check.append((split, obj[split]))
    else:
        to_check.append(("<single>", obj))

    required_turn_keys = {
        "sr_no",
        "history",
        "audio_history",
        "video_history",
        "resp",
        "emotion_label",
        "emotion_token_index",
        "sentiment_token_index",
        "dia_emotion_forecast",
        "user_emotion_forecast",
        "speaker",
        "g",
    }

    for split, split_obj in to_check:
        if not isinstance(split_obj, dict):
            errs.append(f"{split}: must be dict(turn_num -> dialogues)")
            continue

        turn_keys = list(split_obj.keys())
        if not turn_keys:
            errs.append(f"{split}: empty split")
            continue

        turn_key = turn_keys[0]
        dialogues = split_obj[turn_key]
        if not isinstance(dialogues, list) or not dialogues:
            errs.append(f"{split}.{turn_key}: must be non-empty list of dialogues")
            continue

        one_dialogue = dialogues[0]
        if not isinstance(one_dialogue, list) or not one_dialogue:
            errs.append(f"{split}.{turn_key}[0]: must be non-empty list of turns")
            continue

        one_turn = one_dialogue[0]
        if not isinstance(one_turn, dict):
            errs.append(f"{split}.{turn_key}[0][0]: must be dict")
            continue

        miss = sorted(required_turn_keys - set(one_turn.keys()))
        if miss:
            errs.append(f"{split}.{turn_key}[0][0]: missing keys {miss}")

    return errs


def main() -> int:
    args = parse_args()

    try:
        pkl_obj = load_pickle(args.pkl)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    json_obj = load_json(args.json)

    comp = Comparator(args.max_items, args.max_diffs, args.tol)
    comp.compare(pkl_obj, json_obj)

    print(f"Visited nodes: {comp.visited}")
    print(f"Found differences: {len(comp.diffs)}")
    if comp.diffs:
        print("---- Difference Samples ----")
        for d in comp.diffs:
            print(d)

    if args.reader_check:
        print("---- Reader Encoded Shape Check (PKL) ----")
        errs = check_reader_encoded_shape(pkl_obj)
        if errs:
            for e in errs:
                print(f"[FAIL] {e}")
        else:
            print("[PASS] pkl matches reader encoded_data structural expectations.")

    return 1 if comp.diffs else 0


if __name__ == "__main__":
    sys.exit(main())
