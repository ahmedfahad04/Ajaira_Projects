import numpy as np

class StatisticCommand:
    def __init__(self, operation, post_process=None):
        self.operation = operation
        self.post_process = post_process or (lambda x: x)
    
    def execute(self, data):
        result = self.operation(data)
        return self.post_process(result)

class DataStatistics2:
    def __init__(self, data):
        self.data = np.array(data)
        
        # Define commands for each statistic
        commands = {
            'sum': StatisticCommand(np.sum),
            'min': StatisticCommand(np.min),
            'max': StatisticCommand(np.max),
            'variance': StatisticCommand(np.var, lambda x: round(x, 2)),
            'std_deviation': StatisticCommand(np.std, lambda x: round(x, 2)),
            'correlation': StatisticCommand(lambda x: np.corrcoef(x, rowvar=False))
        }
        
        # Dynamically create getter methods
        for stat_name, command in commands.items():
            method_name = f'get_{stat_name}'
            setattr(self, method_name, self._create_getter(command))
    
    def _create_getter(self, command):
        def getter():
            return command.execute(self.data)
        return getter
