# coding:utf-8
from datetime import datetime


def time_delta(t: datetime):
    """ 计算现在到指定时间的间隔

    Parameters
    ----------
    t: datatime
        开始时间

    Returns
    -------
    delta_time: str
        时间间隔
    """
    dt = datetime.now() - t
    hours = dt.seconds//3600
    minutes = (dt.seconds-hours*3600) // 60
    seconds = dt.seconds % 60
    return f'{hours:02}:{minutes:02}:{seconds:02}'

