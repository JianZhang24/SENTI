import pickle
import json
import argparse

def convert(obj):
    """
    将一些无法直接 JSON 序列化的对象转换为可序列化格式
    """
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    raise TypeError(f"Type {type(obj)} not serializable")


def pkl_to_json(pkl_path, json_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=convert, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PKL file to JSON")
    parser.add_argument("input", help="input .pkl file")
    parser.add_argument("output", help="output .json file")

    args = parser.parse_args()

    pkl_to_json(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")