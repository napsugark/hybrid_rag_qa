#!/usr/bin/env python3
"""
Start the API server for Mayan EDMS integration

Usage:
    poetry run python run_api_server.py
    poetry run python run_api_server.py --port 8080
    poetry run python run_api_server.py --reload
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.server import start_server


def main():
    parser = argparse.ArgumentParser(
        description="Start the Hybrid RAG API server for Mayan EDMS integration"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Starting Hybrid RAG API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print("")
    print("Endpoints:")
    print(f"  - Health:  http://{args.host}:{args.port}/health")
    print(f"  - Docs:    http://{args.host}:{args.port}/docs")
    print(f"  - Webhook: http://{args.host}:{args.port}/webhook/mayan")
    print("=" * 60)
    
    start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
