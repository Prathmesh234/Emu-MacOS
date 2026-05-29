from fastapi import WebSocket


class ConnectionManager:
    """
    Fan-out registry of WebSockets per session_id.

    A single agent session can legitimately have MULTIPLE listeners — the
    desktop Electron renderer is the obvious one, but the messaging bridge
    (whatsapp / imessage dispatcher) also subscribes to the shared
    "remote_control" session over its own WS to learn when each turn ends.
    A naive dict[str, WebSocket] silently overwrites the older socket on
    every new subscriber and breaks turn-end propagation for everyone
    except the most recent connect — which manifested as the messaging
    bridge's per-handle queue stalling for the full 5-minute safety-net
    timeout after the FIRST inbound message and never recovering.

    The store is a dict[str, list[WebSocket]] and `send` broadcasts to
    every still-open socket for the session. Sockets that fail to receive
    (closed mid-send) are pruned in place.
    """

    def __init__(self):
        self._sockets: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self._sockets.setdefault(session_id, []).append(ws)
        n = len(self._sockets[session_id])
        suffix = f" subscribers={n}" if n > 1 else ""
        print(f"[ws] connected  session={session_id}{suffix}")

    def disconnect(self, session_id: str, ws: WebSocket | None = None):
        """
        Drop a single subscriber. When `ws` is None (legacy callers),
        falls back to removing the most-recently-attached socket so old
        code paths keep working, but new callers should always pass the
        specific WS instance they're tearing down.
        """
        bucket = self._sockets.get(session_id)
        if not bucket:
            return
        if ws is None:
            bucket.pop()
        else:
            try:
                bucket.remove(ws)
            except ValueError:
                pass
        if not bucket:
            self._sockets.pop(session_id, None)
        print(f"[ws] disconnected session={session_id}")

    async def send(self, session_id: str, message: dict):
        bucket = self._sockets.get(session_id)
        if not bucket:
            return
        # Iterate over a snapshot so prune-on-failure can mutate the list
        # safely. We deliberately fan out best-effort: one slow/closed
        # consumer must NEVER block the others (a closed renderer WS used
        # to silently stall the messaging dispatcher's turn-end signal).
        dead: list[WebSocket] = []
        for ws in list(bucket):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                bucket.remove(ws)
            except ValueError:
                pass
        if not bucket:
            self._sockets.pop(session_id, None)
