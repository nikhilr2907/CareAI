"""
Fleet-level event logging: charging interrupts, robot stoppages, offline events.
Separate from TaskLogger so operational events are not buried in task completion noise.
"""
import logging
from pathlib import Path
from typing import Optional, List


class FleetEventLogger:
    """Log robot-level operational events to a dedicated file.

    Events logged:
        CHARGE_INTERRUPT  - battery forced a mid-task diversion to hub
        CHARGE_DISPATCH   - robot routed to charge while idle (no tasks)
        CHARGE_START      - robot docked and charging began
        CHARGE_COMPLETE   - charging finished, robot resuming
        ROBOT_OFFLINE     - manual shutdown; tasks requeued or escalated
        TASK_REQUEUED     - task returned to pending after offline event
        TASK_EMERGENCY    - task escalated to manual handling (items on robot)
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger('GAPO_Fleet')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []

        fh = logging.FileHandler(self.log_dir / "fleet_events.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)

        # Summary counters
        self.charge_interrupts = 0
        self.charge_dispatches_idle = 0
        self.robot_offline_events = 0
        self.emergency_tasks = 0
        self.requeued_tasks = 0

    def log_charge_interrupt(
        self,
        robot_id: int,
        sim_time: float,
        battery_level: float,
        interrupted_task_id: Optional[int],
        queued_task_ids: List[int],
        current_node: Optional[int],
        hub_node: Optional[int],
    ):
        """Robot was mid-task but battery forced a diversion to charge."""
        self.charge_interrupts += 1
        queued_str = str(queued_task_ids) if queued_task_ids else "[]"
        self.logger.info(
            f"CHARGE_INTERRUPT robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} interrupted_task={interrupted_task_id} "
            f"queued_tasks={queued_str} current_node={current_node} hub_node={hub_node} "
            f"(total_interrupts={self.charge_interrupts})"
        )

    def log_charge_dispatch_idle(
        self,
        robot_id: int,
        sim_time: float,
        battery_level: float,
        current_node: Optional[int],
        hub_node: Optional[int],
    ):
        """Robot was idle with no tasks and dispatched to charge."""
        self.charge_dispatches_idle += 1
        self.logger.info(
            f"CHARGE_DISPATCH_IDLE robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} current_node={current_node} hub_node={hub_node}"
        )

    def log_charge_start(self, robot_id: int, sim_time: float, battery_level: float, hub_node: Optional[int]):
        self.logger.info(
            f"CHARGE_START robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} hub_node={hub_node}"
        )

    def log_charge_complete(self, robot_id: int, sim_time: float, battery_level: float, hub_node: Optional[int]):
        self.logger.info(
            f"CHARGE_COMPLETE robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} hub_node={hub_node}"
        )

    def log_task_failed(
        self,
        robot_id: int,
        sim_time: float,
        task_id: int,
        action: str,
        retry_count: int,
        current_node: Optional[int],
    ):
        """A navigation goal failed/canceled. action is 'retry' or 'abort'."""
        self.logger.info(
            f"TASK_FAILED robot={robot_id} sim_time={sim_time:.1f}s task={task_id} "
            f"action={action} retry_count={retry_count} current_node={current_node}"
        )

    def log_robot_offline(
        self,
        robot_id: int,
        sim_time: float,
        battery_level: float,
        current_node: Optional[int],
        requeued_task_ids: List[int],
        emergency_task_ids: List[int],
    ):
        """Robot manually shut down; tasks either requeued or escalated."""
        self.robot_offline_events += 1
        self.emergency_tasks += len(emergency_task_ids)
        self.requeued_tasks += len(requeued_task_ids)
        self.logger.info(
            f"ROBOT_OFFLINE robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} current_node={current_node} "
            f"requeued={requeued_task_ids} emergency={emergency_task_ids}"
        )

    def log_robot_stationary(
        self,
        robot_id: int,
        sim_time: float,
        battery_level: float,
        duration_s: float,
        current_node: Optional[int],
        stoppage_reason: str,
    ):
        """Robot has been stationary for >= 10s. stoppage_reason populated from scheduler
        state in sim; to be enriched with ROS backend data in real deployment."""
        self.logger.info(
            f"ROBOT_STATIONARY robot={robot_id} sim_time={sim_time:.1f}s "
            f"battery={battery_level:.3f} duration={duration_s:.1f}s "
            f"current_node={current_node} stoppage_reason={stoppage_reason}"
        )

    def log_summary(self):
        self.logger.info(
            f"=== FLEET SUMMARY === "
            f"charge_interrupts={self.charge_interrupts} "
            f"idle_dispatches={self.charge_dispatches_idle} "
            f"offline_events={self.robot_offline_events} "
            f"emergency_tasks={self.emergency_tasks} "
            f"requeued_tasks={self.requeued_tasks}"
        )