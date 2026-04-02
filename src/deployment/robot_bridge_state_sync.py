"""
ROS Bridge Task Dispatcher - Translates CareRobotics tasks to bridge wire format.
"""

import logging
from typing import Optional

from .robot_bridge_websocket_client import RobotBridgeWebSocketClient
from ..environment.graph.graph_state import GraphState


logger = logging.getLogger(__name__)


class RobotBridgeTaskDispatcher:
    """
    Submits tasks to robot via ROS bridge.

    Translates CareRobotics tasks to bridge task format and sends via WebSocket.
    """

    def __init__(self, client: RobotBridgeWebSocketClient, graph_state: Optional[GraphState]):
        """
        Initialize submitter.

        Args:
            client: RobotBridgeWebSocketClient instance
            graph_state: GraphState instance (for node coordinates). Can be set later via set_graph_state().
        """
        self.client = client
        self.graph_state = graph_state

    async def submit_task(self, task, robot_id: int = 0) -> bool:
        """
        Submit a task to the robot via bridge.

        Args:
            task: Task object with planned_path (List[int] of node indices)
            robot_id: Which robot to assign to (default 0)

        Returns:
            True if submitted successfully
        """
        try:
            waypoints = []
            if hasattr(task, 'planned_path') and task.planned_path:
                for node_idx in task.planned_path:
                    node = self.graph_state.nodes[node_idx]
                    waypoints.append({
                        'x': node.center_x,
                        'y': node.center_y,
                        'z': 0.0
                    })
            elif hasattr(task, 'from_location_index') and hasattr(task, 'to_location_index'):
                from_node = self.graph_state.nodes[task.from_location_index]
                to_node = self.graph_state.nodes[task.to_location_index]
                waypoints = [
                    {'x': from_node.center_x, 'y': from_node.center_y, 'z': 0.0},
                    {'x': to_node.center_x, 'y': to_node.center_y, 'z': 0.0}
                ]

            if not waypoints:
                logger.error(f"No waypoints for task {task.task_id}")
                return False

            task_command = {
                'task_id': task.task_id,
                'robot_id': robot_id,
                'waypoints': waypoints,
                'task_metadata': {
                    'task_type': getattr(task, 'task_type', 'unknown'),
                    'num_items': getattr(task, 'num_items', 1),
                    'sku_id': getattr(task, 'sku_id', None),
                    'description': f"Task {task.task_id}"
                }
            }

            success = await self.client.send_task_command(task_command)
            if success:
                logger.info(f"Submitted task {task.task_id} to robot {robot_id}")
            return success

        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return False

    async def cancel_task(self, task_id: int) -> bool:
        """Cancel a task on the robot."""
        return await self.client.send_task_cancel(task_id)
