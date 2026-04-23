"""
SKU state logger — writes a time-series CSV of inventory levels for post-run analytics.

Two record types share the same schema:

  SNAPSHOT  — periodic full-state sweep of every SKU at every node
  RESTOCK   — point event when a robot dropoff increases stock
  STOCKOUT  — point event when a SKU first hits zero
  REORDER   — point event when stock crosses below the reorder threshold

CSV columns (analytics-ready, one row per SKU per event):
  sim_time, event_type, node_idx, node_tag, floor,
  sku_id, category, stock, max_stock, par, reorder,
  fill_pct, below_reorder, is_stockout, consumption_rate_per_hour
"""

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_COLUMNS = [
    "sim_time",
    "event_type",
    "node_idx",
    "node_tag",
    "floor",
    "sku_id",
    "category",
    "stock",
    "max_stock",
    "par",
    "reorder",
    "fill_pct",
    "below_reorder",
    "is_stockout",
    "consumption_rate_per_hour",
]


class SKULogger:
    """Logs SKU inventory state to a CSV file for post-run analytics.

    Usage:
        sku_logger = SKULogger(log_dir, snapshot_interval_s=60.0)
        # each env step:
        sku_logger.maybe_snapshot(graph_state, sim_time)
        # on robot dropoff:
        sku_logger.log_restock(node_idx, node_tag, floor, sku_id, category,
                               stock_before, stock_after, max_stock, par, reorder, sim_time)
    """

    def __init__(self, log_dir: Path, snapshot_interval_s: float = 60.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval_s = snapshot_interval_s

        self._csv_path = self.log_dir / "sku_states.csv"
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        # Track state to detect edge-triggered events
        self._last_snapshot_time: float = -1.0
        self._stockout_active: dict = {}   # (node_idx, sku_id) -> bool
        self._below_reorder_active: dict = {}  # (node_idx, sku_id) -> bool

        logger.info(f"SKULogger writing to {self._csv_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_snapshot(self, graph_state, sim_time: float):
        """Write a SNAPSHOT row for every SKU at every node if the interval has elapsed."""
        if sim_time - self._last_snapshot_time < self.snapshot_interval_s:
            return
        self._last_snapshot_time = sim_time
        self._write_full_snapshot(graph_state, sim_time)

    def log_restock(
        self,
        node_idx: int,
        node_tag: str,
        floor,
        sku_id: str,
        category: str,
        stock_after: float,
        max_stock: float,
        par: float,
        reorder: float,
        sim_time: float,
        consumption_rate: float = 0.0,
    ):
        """Write a RESTOCK row when a robot dropoff increases stock."""
        self._write_row(
            event_type="RESTOCK",
            sim_time=sim_time,
            node_idx=node_idx,
            node_tag=node_tag,
            floor=floor,
            sku_id=sku_id,
            category=category,
            stock=stock_after,
            max_stock=max_stock,
            par=par,
            reorder=reorder,
            consumption_rate=consumption_rate,
        )

    def close(self):
        """Flush and close the CSV file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info(f"SKULogger closed: {self._csv_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_full_snapshot(self, graph_state, sim_time: float):
        """Iterate all nodes and write one row per SKU."""
        for node_idx, node in enumerate(graph_state.nodes):
            if not node.sku_inventory:
                continue
            node_tag = getattr(node, "location_tag", None) or f"node_{node_idx}"
            floor = getattr(node, "floor", None)
            for sku_id, sku_data in node.sku_inventory.items():
                stock = float(sku_data.get("stock", 0.0))
                max_stock = float(sku_data.get("max", 0.0))
                par = float(sku_data.get("par", 0.0))
                reorder = float(sku_data.get("reorder", 0.0))
                category = sku_data.get("category", "")

                # Consumption rate from cached value if available
                rate = float(sku_data.get("rate", 0.0))

                self._write_row(
                    event_type="SNAPSHOT",
                    sim_time=sim_time,
                    node_idx=node_idx,
                    node_tag=node_tag,
                    floor=floor,
                    sku_id=sku_id,
                    category=category,
                    stock=stock,
                    max_stock=max_stock,
                    par=par,
                    reorder=reorder,
                    consumption_rate=rate,
                )

                # Edge-triggered STOCKOUT event
                key = (node_idx, sku_id)
                was_stockout = self._stockout_active.get(key, False)
                now_stockout = stock <= 0.0
                if now_stockout and not was_stockout:
                    self._write_row(
                        event_type="STOCKOUT",
                        sim_time=sim_time,
                        node_idx=node_idx,
                        node_tag=node_tag,
                        floor=floor,
                        sku_id=sku_id,
                        category=category,
                        stock=stock,
                        max_stock=max_stock,
                        par=par,
                        reorder=reorder,
                        consumption_rate=rate,
                    )
                self._stockout_active[key] = now_stockout

                # Edge-triggered REORDER event
                was_below = self._below_reorder_active.get(key, False)
                now_below = stock <= reorder
                if now_below and not was_below:
                    self._write_row(
                        event_type="REORDER",
                        sim_time=sim_time,
                        node_idx=node_idx,
                        node_tag=node_tag,
                        floor=floor,
                        sku_id=sku_id,
                        category=category,
                        stock=stock,
                        max_stock=max_stock,
                        par=par,
                        reorder=reorder,
                        consumption_rate=rate,
                    )
                self._below_reorder_active[key] = now_below

        self._file.flush()

    def _write_row(
        self,
        event_type: str,
        sim_time: float,
        node_idx: int,
        node_tag: str,
        floor,
        sku_id: str,
        category: str,
        stock: float,
        max_stock: float,
        par: float,
        reorder: float,
        consumption_rate: float,
    ):
        fill_pct = round(stock / max_stock * 100, 2) if max_stock > 0 else 0.0
        self._writer.writerow({
            "sim_time": round(sim_time, 2),
            "event_type": event_type,
            "node_idx": node_idx,
            "node_tag": node_tag,
            "floor": floor if floor is not None else "",
            "sku_id": sku_id,
            "category": category,
            "stock": round(stock, 4),
            "max_stock": round(max_stock, 4),
            "par": round(par, 4),
            "reorder": round(reorder, 4),
            "fill_pct": fill_pct,
            "below_reorder": int(stock <= reorder),
            "is_stockout": int(stock <= 0.0),
            "consumption_rate_per_hour": round(consumption_rate, 6),
        })