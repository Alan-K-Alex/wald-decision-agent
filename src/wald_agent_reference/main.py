from __future__ import annotations

import argparse
import json

from .core.agent import LeadershipInsightAgent
from .evaluation.evaluate import EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wald Agent Reference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Answer a leadership question")
    ask_parser.add_argument("--docs", required=True, help="Path to the document folder")
    ask_parser.add_argument("--question", required=True, help="Leadership question to answer")
    ask_parser.add_argument("--plot", action="store_true", help="Force plot generation when possible")

    eval_parser = subparsers.add_parser("evaluate", help="Run validation set")
    eval_parser.add_argument("--docs", required=True, help="Path to the document folder")
    eval_parser.add_argument("--validation", required=True, help="Path to validation json file")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent = LeadershipInsightAgent()

    if args.command == "ask":
        response = agent.ask(question=args.question, docs_path=args.docs, generate_plot=args.plot)
        print(response.to_markdown())
        return

    if args.command == "evaluate":
        runner = EvaluationRunner(agent)
        results = runner.run(docs_path=args.docs, validation_path=args.validation)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
