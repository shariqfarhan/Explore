class MemoryManager:
    def __init__(self):
        self.conversation_history = []

    def add_to_memory(self, interaction):
        self.conversation_history.append(interaction)

    def get_memory(self):
        return self.conversation_history
