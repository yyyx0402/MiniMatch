import argparse
import json
import socket
from urllib.request import urlopen

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run the local MiniMatch application")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.settimeout(.3)
        occupied = probe.connect_ex(("127.0.0.1", args.port)) == 0
    if occupied:
        try:
            with urlopen(f"http://127.0.0.1:{args.port}/api/application", timeout=2) as response:
                info = json.load(response)
            if info.get("default_search") == "hybrid" and info.get("local_only") is True:
                print(f"MiniMatch is already running: http://127.0.0.1:{args.port}/")
                return
        except (OSError, ValueError):
            pass
        parser.exit(1, f"Port {args.port} is in use. Choose another port with --port.\n")
    uvicorn.run("productsearch.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
