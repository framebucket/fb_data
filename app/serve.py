"""Start the API on one dual-stack socket (IPv4 + IPv6) so both Railway's public proxy (IPv4) and its private
network (IPv6) can connect. `uvicorn --host ::` may bind IPv6-only, which makes the proxy fail with
"connection refused".

    python -m app.serve            # listens on $PORT (default 8001)
"""
from __future__ import annotations

import os
import socket

import uvicorn


def dual_stack_socket(port: int) -> socket.socket:
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        address: tuple = ("::", port)
    except OSError:                       # no IPv6 in this container: plain IPv4
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        address = ("0.0.0.0", port)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(address)
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def main() -> None:
    port = int(os.getenv("PORT", "8001"))
    sock = dual_stack_socket(port)
    config = uvicorn.Config("app.api:app", proxy_headers=True, forwarded_allow_ips="*", log_level="info")
    stack = "IPv4 + IPv6" if sock.family == socket.AF_INET6 else "IPv4"
    print(f"FrameBucket data service listening on port {port} ({stack})", flush=True)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()
