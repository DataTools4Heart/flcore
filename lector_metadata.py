#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter


def describe_structure(obj, indent=0, max_list_examples=3):
    """
    Describe recursivamente la estructura de un objeto JSON.
    """
    prefix = " " * indent

    if isinstance(obj, dict):
        lines = [f"{prefix}dict ({len(obj)} keys)"]
        for key, value in obj.items():
            lines.append(f"{prefix}├── {key}:")
            lines.extend(
                describe_structure(
                    value,
                    indent=indent + 4,
                    max_list_examples=max_list_examples
                )
            )
        return lines

    elif isinstance(obj, list):
        lines = [f"{prefix}list ({len(obj)} items)"]

        if not obj:
            lines.append(f"{prefix}└── empty")
            return lines

        # Analizar algunos ejemplos
        examples = obj[:max_list_examples]

        types = Counter(type(x).__name__ for x in obj)
        lines.append(
            f"{prefix}└── item types: {dict(types)}"
        )

        for i, item in enumerate(examples):
            lines.append(f"{prefix}    example[{i}]:")
            lines.extend(
                describe_structure(
                    item,
                    indent=indent + 8,
                    max_list_examples=max_list_examples
                )
            )

        if len(obj) > max_list_examples:
            lines.append(
                f"{prefix}    ... ({len(obj)-max_list_examples} more items)"
            )

        return lines

    else:
        return [f"{prefix}{type(obj).__name__}"]


def analyze_metadata_file(metadata_path):
    print("=" * 80)
    print(f"Directory: {metadata_path.parent}")
    print(f"File: {metadata_path.name}")
    print("-" * 80)

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        structure = describe_structure(data)
        print("\n".join(structure))

    except Exception as e:
        print(f"ERROR reading file: {e}")

    print()


def main(root_dir="."):
    root = Path(root_dir)

    metadata_files = sorted(root.rglob("metadata.json"))

    if not metadata_files:
        print("No metadata.json files found.")
        return

    print(f"Found {len(metadata_files)} metadata.json files\n")

    for metadata_file in metadata_files:
        analyze_metadata_file(metadata_file)


if __name__ == "__main__":
    main("/home/dt4h/CDM_tools/feature-extraction-suite/output-data/myFhirServer/dataset/study1-fs")
