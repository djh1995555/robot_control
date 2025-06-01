class DataLogger():
    def __init__(self):
        self.data_dict = {}
    
    def add_data(self, data_name, data):
        if data_name not in self.data_dict.keys():
            self.data_dict[data_name] = []
        self.data_dict[data_name].append(data)

    def fill_lost_data(self):
        timestamp_len = len(self.data_dict['timestamp'])
        for data_name, data in self.data_dict.items():
            if(timestamp_len != len(data)):
                if(timestamp_len - len(data) == 1):
                    self.data_dict[data_name].append(None)
                else:
                    print(f'{data_name} data has wrong!')
    
    def get_data(self):
        return self.data_dict