"""
Digital Twin WebSocket publisher.

Runs two servers in background threads:
  - WebSocket on `port`   (default 8766) — pushes robot state to browser
  - HTTP      on `port+1` (default 8767) — serves digital_twin/ static files

Usage:
    publisher = TwinPublisher(graph_state=env.graph_state, port=8766)
    publisher.start()

    # Inside loop:
    publisher.broadcast(env)

    publisher.stop()  # called automatically on KeyboardInterrupt via finally block
"""

import asyncio
import json
import logging
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

_DIGITAL_TWIN_DIR = Path(__file__).parent.parent.parent / "digital_twin"


class TwinPublisher:
    def __init__(self, graph_state, port: int = 8766):
        self.port = port
        self._graph_state = graph_state
        self._clients: Set = set()
        self._latest_payload: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._http_server: Optional[HTTPServer] = None

    def start(self) -> None:
        """Write graph config and start both servers in background threads."""
        _DIGITAL_TWIN_DIR.mkdir(exist_ok=True)
        self._write_graph_json()

        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()

        logger.info(
            f"Digital twin: ws://localhost:{self.port}  |  "
            f"http://localhost:{self.port + 1}"
        )

    def stop(self) -> None:
        if self._http_server:
            self._http_server.shutdown()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def broadcast(self, env, robot_backend=None) -> None:
        """Serialize env state and push to all connected browser clients."""
        payload = json.dumps(self._build_payload(env, robot_backend=robot_backend))
        self._latest_payload = payload
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(
                self._push_to_all(payload), self._loop
            )

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    def _build_payload(self, env, robot_backend=None) -> dict:
        robots = []
        sims = getattr(env, "robot_simulators", []) or []

        # Sim mode: robot_simulators are live objects
        sim_robots = [(robot, sim) for robot, sim in zip(env.robots, sims) if sim is not None]
        if sim_robots:
            for robot, sim in sim_robots:
                robots.append({
                    "id": robot.robot_id,
                    "x": float(sim.x),
                    "y": float(sim.y),
                    "heading": float(sim.heading),
                    "velocity": float(sim.velocity_ms),
                    "battery": float(sim.battery_level),
                    "task_id": sim.active_task_id,
                    "is_charging": bool(sim.is_charging),
                    "edge_progress": float(sim.edge_progress),
                })
        elif robot_backend is not None:
            # Real deployment mode: pull latest telemetry from bridge
            for tel in robot_backend.get_all_telemetry():
                robots.append({
                    "id": tel.robot_id,
                    "x": float(tel.x),
                    "y": float(tel.y),
                    "heading": float(tel.heading),
                    "velocity": float(tel.velocity_ms),
                    "battery": float(tel.battery_level),
                    "task_id": tel.active_task_id,
                    "is_charging": False,
                    "edge_progress": 0.0,
                })

        tasks = [
            {
                "id": t.task_id,
                "from": getattr(t, "from_location_index", None),
                "to": getattr(t, "to_location_index", None),
                "status": "pending",
            }
            for t in list(env.pending_tasks)[:20]
        ]

        return {
            "t": float(env.current_time),
            "robots": robots,
            "tasks": tasks,
        }

    # ------------------------------------------------------------------
    # WebSocket server
    # ------------------------------------------------------------------

    async def _ws_handler(self, websocket) -> None:
        self._clients.add(websocket)
        logger.debug(f"Twin client connected ({len(self._clients)} total)")
        try:
            if self._latest_payload:
                await websocket.send(self._latest_payload)
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.debug(f"Twin client disconnected ({len(self._clients)} remaining)")

    async def _push_to_all(self, message: str) -> None:
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def _run_ws_server(self) -> None:
        import websockets

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _serve():
            async with websockets.serve(self._ws_handler, "0.0.0.0", self.port):
                await asyncio.Future()

        try:
            self._loop.run_until_complete(_serve())
        except Exception as e:
            logger.error(f"Twin WS server error: {e}")

    # ------------------------------------------------------------------
    # HTTP server (serves digital_twin/ static files)
    # ------------------------------------------------------------------

    def _run_http_server(self) -> None:
        os.chdir(_DIGITAL_TWIN_DIR)

        class _Handler(SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence per-request logs

        http_port = self.port + 1
        self._http_server = HTTPServer(("0.0.0.0", http_port), _Handler)
        self._http_server.serve_forever()

    # ------------------------------------------------------------------
    # Graph JSON (written once at startup, read by index.html)
    # ------------------------------------------------------------------

    def _write_graph_json(self) -> None:
        nodes = []
        for i, node in enumerate(self._graph_state.nodes):
            nodes.append({
                "index": i,
                "id": node.node_id,
                "name": getattr(node, "name", node.node_id),
                "type": getattr(node, "node_type", "unknown"),
                "x": float(node.x),
                "y": float(node.y),
            })

        edges = []
        for edge in self._graph_state.edges:
            edges.append({
                "from": edge.from_node,
                "to": edge.to_node,
                "distance_m": float(edge.distance_m),
            })

        graph = {"nodes": nodes, "edges": edges}
        out = _DIGITAL_TWIN_DIR / "graph.json"
        out.write_text(json.dumps(graph, indent=2))
        logger.debug(f"Graph JSON written to {out}")
