# coding: utf-8
import os
import tarfile

import requests


class Downloader:
    """ 数据集下载器 """

    def __init__(self, urls: list):
        """
        Parameters
        ----------
        urls: list
            下载链接
        """
        if not isinstance(urls, list):
            raise ValueError("urls 必须是列表")

        self.urls = urls

    def download(self, chunk_size=4, redownload=False, unzip=True):
        """ 下载所有数据集

        Parameters
        ----------
        chunk_size: int
            分块大小，以 MB 为单位

        redownload: bool
            如果文件夹下以及存在了同名文件，是否重新下载

        unzip: bool
            下载完是否自动解压，只支持解压 .tar 文件
        """
        filenames = []
        for url in self.urls:
            file = url.split('/')[-1]  # type:str

            if os.path.exists(file) and not redownload:
                continue

            filenames.append(file)
            print(f'正在下载 {file} ...')

            # 分块下载
            response = requests.get(url, stream=True)
            with open(file, 'wb') as f:
                for chunk in response.iter_content(chunk_size*1024):
                    if chunk:
                        f.write(chunk)

            print(f'完成下载 {file}')

        # 解压文件
        if unzip:
            for file in filenames:
                if not file.endswith('.tar'):
                    continue
                
                print(f'正在解压 {file} ...')
                tar = tarfile.open(file)
                tar.extractall(file.split('.')[0])
                print(f'完成解压 {file}')


if __name__ == '__main__':
    downloader = Downloader([
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    ])

    downloader.download(chunk_size=10, redownload=False)
