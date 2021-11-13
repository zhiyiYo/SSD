# coding:utf-8
import os
import json


class LossLogger:
    """ 损失记录器 """

    def __init__(self, frequency: int, log_file: str = None, save_dir='log'):
        """
        Parameters
        ----------
        frequency: int
            记录数据的频率

        log_file: str
            损失数据历史记录文件，要求是 json 文件

        save_dir: str
            损失数据保存的文件夹
        """
        self.frequency = frequency
        self.log_file = log_file
        self.save_dir = save_dir
        self.loss = 0
        self.loc_loss = 0
        self.conf_loss = 0
        self.losses = []
        self.loc_losses = []
        self.conf_losses = []
        self.n_steps = 0

        # 载入历史数据
        if log_file:
            self.load(log_file)

    def update(self, loc_loss: float, conf_loss: float):
        """ 更新损失 """
        self.loc_loss += loc_loss
        self.conf_loss += conf_loss
        self.loss += (loc_loss+conf_loss)

        # 如果遇到了记录点就记录数据
        self.n_steps += 1
        if self.n_steps % self.frequency == 0:
            self.record()

    def record(self):
        """ 记录一条数据 """
        self.losses.append(self.loss/self.frequency)
        self.loc_losses.append(self.loc_loss/self.frequency)
        self.conf_losses.append(self.conf_loss/self.frequency)
        self.loss = 0
        self.loc_loss = 0
        self.conf_loss = 0

    def load(self, file_path: str):
        """ 载入历史记录数据 """
        if not os.path.exists(file_path):
            raise FileNotFoundError("损失历史纪录文件不存在，请检查文件路径！")

        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)
                self.losses = data['losses']  # type:list
                self.loc_losses = data['loc_losses']  # type:list
                self.conf_losses = data['conf_losses']  # type:list
        except:
            raise Exception("json 文件损坏，无法正确读取内容！")

    def save(self, file_name: str):
        """ 保存记录的数据

        Parameters
        ----------
        file_name: str
            文件名，不包含 `.json` 后缀
        """
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, file_name+'.json'), 'w', encoding='utf-8') as f:
            data = {
                "losses": self.losses,
                "loc_losses": self.loc_losses,
                "conf_losses": self.conf_losses
            }
            json.dump(data, f)
