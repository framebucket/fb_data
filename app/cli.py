from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from .config import Settings
from .service import PhotoService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process photos from the FrameBucket local data_output folder."
    )
    parser.add_argument("--input", type=str, help="Input directory; defaults to data_output/input.")
    parser.add_argument("--event-id", default="LOCAL_EVENT")
    parser.add_argument("--photographer-id", default="LOCAL_PHOTOGRAPHER")
    parser.add_argument("--customer-id", default="LOCAL_CUSTOMER")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    settings = Settings.from_env()
    input_dir = Path(args.input).expanduser() if args.input else settings.input_dir
    service = PhotoService(settings)
    results = service.process_directory(
        input_dir,
        args.event_id,
        args.photographer_id,
        args.customer_id,
        args.workers,
    )
    summary = Counter(result.get("status", "UNKNOWN") for result in results)
    report = {
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(settings.data_root),
        "processed": len(results),
        "ready": summary["READY"],
        "failed": summary["PROCESSING_FAILED"],
        "results": results,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Input: {report['input_directory']}")
        print(f"Output: {report['output_directory']}")
        print(f"Photos found: {report['processed']}; ready: {report['ready']}; failed: {report['failed']}")
        for result in results:
            print(
                f"{result.get('status', 'UNKNOWN'):>18}  "
                f"{result.get('original_filename', result.get('photo_id', ''))}"
            )
            if result.get("error"):
                print(f"  {result['error']}")
        print("Customer previews: events/<event-id>/photos/")
        print("Private originals: photographer/<photographer-id>/<event-id>/originals/")
    return 1 if summary["PROCESSING_FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
