import json
import numpy as np
import time

class DataFrame(object):
    """[summary]

        Args:
            current_time_step (int): [description]
            current_time (float): [description]
            data (dict): [description]
        """

    def __init__(self, current_time_step: int, current_time: float, data: dict) -> None:
        self.current_time_step = current_time_step
        self.current_time = current_time
        self.data = data

    def get_dict(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        return {"current_time": self.current_time, "current_time_step": self.current_time_step, "data": self.data}

    def __str__(self) -> str:
        return str(self.get_dict())

    @classmethod
    def init_from_dict(cls, dict_representation: dict):
        """[summary]

        Args:
            dict_representation (dict): [description]

        Returns:
            DataFrame: [description]
        """
        frame = object.__new__(cls)
        frame.current_time_step = dict_representation["current_time_step"]
        frame.current_time = dict_representation["current_time"]
        frame.data = dict_representation["data"]
        return frame

class DataRecorder:
    def __init__(self):
        self.data = {
            "name": "residual_pushing"
        }
        return
    
    def set_new_frame(self, frame_id, time, time_step, data):
        self.data[frame_id] = { 
                    "current_time": time, 
                    "time_step": time_step,
                    "data": data
                    }

    def save_episode_file(self, file_path):
        with open(file_path, "w") as outfile:
            json.dump(self.data, outfile)
    
    def clear_dict(self) -> None:
        del self.data
    
    def reset_dict(self) -> None:
        name = "residual_pushing"
        self.data = {
            "name": name
        }
        

