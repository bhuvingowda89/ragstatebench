from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_campaign, validate_clean


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ragstate")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate-clean", "validate-controls"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--campaign", default="pilot")
        if command != "validate-clean":
            p.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    # JSON is a YAML 1.2 subset; the bundled .yaml needs no parser dependency.
    config = json.loads(Path(args.config).read_text())
    if args.campaign != config.get("campaign_id", "pilot"):
        parser.error("--campaign differs from config campaign_id")
    if args.command == "validate-clean":
        result = validate_clean(config)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["pass"] else 1
    output = args.output or config.get("output", "results/pilot")
    rows = run_campaign(config, output)
    if args.command == "validate-controls":
        from .validation import validate_controls
        result = validate_controls(rows)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["pass"] else 1
    print(json.dumps({"rows": len(rows), "output": output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
