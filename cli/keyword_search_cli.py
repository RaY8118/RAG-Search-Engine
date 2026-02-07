import argparse

from lib.keyword_search import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25 algorithm"
    )

    search_parser.add_argument("query", help="Search query")

    search_parser = subparsers.add_parser("build", help="Build")

    search_parser = subparsers.add_parser("tf")
    search_parser.add_argument("doc_id", type=int, help="Document ID for to check")
    search_parser.add_argument("term", help="Search term to find counts for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i} {result['title']}")

        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
