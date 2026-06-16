
import json
from pathlib import Path
from collections import Counter


def describe(obj, depth=0, max_depth=4, indent=0):
    prefix = " " * indent

    if depth >= max_depth:
        return [f"{prefix}... (max depth reached)"]

    if isinstance(obj, dict):
        lines = [f"{prefix}dict ({len(obj)} keys)"]
        for k, v in obj.items():
            lines.append(f"{prefix}├── {k}:")
            lines.extend(describe(v, depth + 1, max_depth, indent + 4))
        return lines

    elif isinstance(obj, list):
        lines = [f"{prefix}list ({len(obj)} items)"]

        if not obj:
            return lines + [f"{prefix}└── empty"]

        types = Counter(type(x).__name__ for x in obj)
        lines.append(f"{prefix}└── item types: {dict(types)}")

        # solo primeros ejemplos
        for i, item in enumerate(obj[:3]):
            lines.append(f"{prefix}    example[{i}]:")
            lines.extend(describe(item, depth + 1, max_depth, indent + 8))

        return lines

    else:
        return [f"{prefix}{type(obj).__name__}"]


def analyze_file(path):
    print("=" * 60)
    print(path)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print("\n".join(describe(data)))
    except Exception as e:
        print("ERROR:", e)


def main(root="."):
    root = Path(root)
    files = root.rglob("metadata.json")

    for f in files:
        analyze_file(f)


if __name__ == "__main__":
    main("/home/dt4h/CDM_tools/feature-extraction-suite/output-data/myFhirServer/dataset/study1-fs")
