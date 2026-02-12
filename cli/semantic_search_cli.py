import argparse

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads properply")

    embed_parsers = subparsers.add_parser(
        "embed_text", help="Embed text with embedding model"
    )
    embed_parsers.add_argument("text", type=str, help="Text to be encoded")

    embed_parsers = subparsers.add_parser(
        "verify_embeddings", help="Verify the embedding model loads properply"
    )

    embed_parsers = subparsers.add_parser(
        "embedquery", help="Encode query with embedding model"
    )
    embed_parsers.add_argument("query", type=str, help="User query to be encoded")
    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
