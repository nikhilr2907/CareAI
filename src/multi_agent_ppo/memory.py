class Memory:
    """Memory buffer for storing robot experiences."""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.global_states = []  # For critic (global state)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.global_states[:]


class MultiAgentMemory:
    """Memory manager for multi-agent system (5 robots)."""

    def __init__(self, num_robots=5):
        self.num_robots = num_robots
        self.memories = [Memory() for _ in range(num_robots)]

    def get_memory(self, robot_id):
        """Get memory buffer for a specific robot."""
        return self.memories[robot_id]

    def clear_all(self):
        """Clear all robot memories."""
        for memory in self.memories:
            memory.clear_memory()

    def __getitem__(self, robot_id):
        """Allow indexing: memories[robot_id]"""
        return self.memories[robot_id]
