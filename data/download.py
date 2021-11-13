# coding: utf-8
import os
import tarfile
from urllib.request import urlretrieve


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

    def download(self, redownload=False, unzip=True):
        """  下载所有数据集

        Parameters
        ----------
        redownload: bool
            如果文件夹下以及存在了同名文件，是否重新下载

        unzip: bool
            下载完是否自动解压，只支持解压 .tar 文件
        """
        def reporthook(n, size, total):
            print(f'\r下载进度：{n*size/total:.0%}', end='')

        filenames = []

        for url in self.urls:
            file = url.split('/')[-1]  # type:str

            if os.path.exists(file) and not redownload:
                continue

            filenames.append(file)
            print(f'正在下载 {file} ...')
            urlretrieve(url, file, reporthook)
            print(f'\n✔️  完成下载 {file}')

        # 解压文件
        if not unzip:
            return

        for file in filenames:
            if not file.endswith('.tar'):
                continue

            print(f'\n正在解压 {file} ...')
            tar = tarfile.open(file)
            tar.extractall(file.split('.')[0])
            print(f'✔️  完成解压 {file}')


if __name__ == '__main__':
    downloader = Downloader([
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    ])

    downloader.download(redownload=False)
