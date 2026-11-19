"""Verify a published model on its synthetic inference example."""

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path


def verify_checksums(assets, filenames):
    digests = {}
    for line in (assets / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if name in digests:
            raise ValueError(f"Duplicate checksum for {name}")
        digests[name] = digest

    for filename in filenames:
        with (assets / filename).open("rb") as asset:
            actual = hashlib.file_digest(asset, "sha256").hexdigest()
        if digests.get(filename) != actual:
            raise ValueError(f"Missing or mismatched SHA-256 for {filename}")
        print(f"SHA-256 verified for {filename}", flush=True)


def compare_row(actual, expected, location, *, csv_output):
    if not isinstance(actual, dict) or actual.keys() != expected.keys():
        raise ValueError(f"{location}: fields differ from the expected output")

    for key, reference in expected.items():
        value = actual[key]
        field = f"{location}.{key}"
        if isinstance(reference, dict):
            compare_row(value, reference, field, csv_output=csv_output)
        elif type(reference) in (int, float):
            if csv_output and isinstance(value, str):
                value = float(value)
            if type(value) not in (int, float) or not math.isclose(
                value, reference, rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError(f"{field}: expected {reference}, got {value!r}")
        elif value != reference or type(value) is not type(reference):
            raise ValueError(f"{field}: expected {reference!r}, got {value!r}")


def verify_release(contract, source, assets, model):
    schema = json.loads((contract / "schema.json").read_text(encoding="utf-8"))
    expected = json.loads(
        (contract / "expected-output.json").read_text(encoding="utf-8")
    )
    assets = assets.resolve()
    sample = assets / "sample-input.csv"
    verify_checksums(assets, (model, sample.name))

    with sample.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=schema["input"]["delimiter"])
        if reader.fieldnames != schema["input"]["columns"]:
            raise ValueError(
                "Sample input column names or order differ from schema.json"
            )
        input_rows = sum(1 for _ in reader)

    process = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            schema["predict_command"],
            "--model",
            str(assets / model),
            "--data",
            str(sample),
        ],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    output_schema = schema["output"]
    csv_output = output_schema["format"] == "csv"
    if csv_output:
        reader = csv.DictReader(io.StringIO(process.stdout))
        if reader.fieldnames != output_schema["columns"]:
            raise ValueError("Prediction CSV columns differ from schema.json")
        actual = list(reader)
    elif output_schema["format"] == "json":
        actual = json.loads(process.stdout)
    else:
        raise ValueError(f"Unsupported prediction format {output_schema['format']!r}")

    if not isinstance(actual, list) or not (
        0 < input_rows == len(actual) == len(expected)
    ):
        raise ValueError("Expected one prediction per sample input row")
    for index, (row, reference) in enumerate(
        zip(actual, expected, strict=True), start=1
    ):
        if not isinstance(reference, dict) or reference.keys() != set(
            output_schema["columns"]
        ):
            raise ValueError(f"Expected output row {index} conflicts with schema.json")
        compare_row(row, reference, f"Row {index}", csv_output=csv_output)
    print(f"Verified {input_rows} prediction(s) from the released model.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    try:
        verify_release(args.contract, args.source, args.assets, args.model)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Model release verification failed: {error}\n")


if __name__ == "__main__":
    main()
