
from PyQt5.QtGui import QColor
from  PyQt5 import  QtCore
import os
import yaml
import logging
import tqdm
import xml.etree.ElementTree
import numpy as np
import pandas as pd
import cv2




def Qlogging(Text_browser_object, message=" ", type='info'):
    black = QColor(0, 0, 0)
    red = QColor(255, 0, 0)
    green = QColor(0, 100, 0)

    if (type.lower()  in ['info' , 'green', 'g'] ):
        Text_browser_object.setTextColor(green)
        Text_browser_object.append(message)
    elif (type.lower() in ['red' , 'r' , 'error']):
        Text_browser_object.setTextColor(red)
        Text_browser_object.append(message)
        Text_browser_object.setTextColor(black)
    elif  (type.lower()  in ['b' , 'black']):
        Text_browser_object.setTextColor(black)
        Text_browser_object.append(message)
        Text_browser_object.setTextColor(black)



def import_yaml(path):
    yaml.warnings({'YAMLLoadWarning': False})
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    # for section in cfg:
    #     print(section)
    return cfg

def check_VOC(root):
    if(os.path.exists(os.path.join(root,'VOCdevkit'))):
        return True
    else:
        return False

def load_annotation(path, category_index):

    tree = xml.etree.ElementTree.parse(path)
    # category_index = list(range(20))
    yx_min = []
    yx_max = []
    cls = []
    difficult = []

    for obj in tree.findall('object'):
        try:
            cls.append(category_index[obj.find('name').text])
        except KeyError:
            continue
        bbox = obj.find('bndbox')
        ymin = int(bbox.find('ymin').text)
        xmin = int(bbox.find('xmin').text)
        ymax = int(bbox.find('ymax').text)
        xmax = int(bbox.find('xmax').text)
        assert ymin < ymax
        assert xmin < xmax
        yx_min.append((ymin, xmin))
        yx_max.append((ymax, xmax))
        difficult.append(int(obj.find('difficult').text))
    size = tree.find('size')
    return tree.find('filename').text, (int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)), yx_min, yx_max, cls, difficult


def Read_VOC(cfg):
    root=os.path.abspath(cfg['dataset']['path'])
    category_index = dict([(name.strip(), i) for i, name in enumerate(cfg['dataset']['category'].split(','))])

    # category_index = list(cfg['category_index'].split(','))
    # category_index = list(range(20))
    # category_index = list(category_index.split(','))[0]
    data = []
    root=os.path.join(root,'VOCdevkit','VOC2007')
    path = os.path.join(root, 'ImageSets', 'Main', cfg['dataset']['phase']) + '.txt'
    if not os.path.exists(path):
        logging.warning(path + ' not exists')

    # path='C:/Users/alibh/Desktop/My_Qt/Paper_2_PyQT_Pytorch/Feature_ext/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'


    with open(path, 'r') as f:
        filenames = [line.strip() for line in f]

    for filename in tqdm.tqdm(filenames):
        filename, size, yx_min, yx_max, cls, difficult = load_annotation(
            os.path.join(root, 'Annotations', filename + '.xml'), category_index)
        if len(cls) <= 0:
            continue
        path = os.path.join(root, 'JPEGImages', filename)
        yx_min = np.array(yx_min, dtype=np.float32)
        yx_max = np.array(yx_max, dtype=np.float32)
        cls = np.array(cls, dtype=np.int)
        difficult = np.array(difficult, dtype=np.uint8)
        assert len(yx_min) == len(cls)
        assert yx_min.shape == yx_max.shape
        assert len(yx_min.shape) == 2 and yx_min.shape[-1] == 2
        data.append(dict(path=path, yx_min=yx_min, yx_max=yx_max, cls=cls, difficult=difficult))
    logging.info('%d of %d images are saved' % (len(data), len(filenames)))
    return data ,category_index

class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df = pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError,):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError,):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()




def calculate_area(row):
    cc=row['yx_max'] - row['yx_min']

    return cc

def get_image_size(row):
    size=cv2.imread(row['path']).shape[0:2]

    return list(size)



class pandasModel_2(QtCore.QAbstractTableModel):

    def __init__(self, data):
        QtCore.QAbstractTableModel.__init__(self)
        self._data =  pd.DataFrame(data)

    def rowCount(self, parent=None):
        return len(self._data.index)

    def columnCount(self, parnet=None):
        return len(self._data.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[col]
        return None

    def sort(self, column, order):
        # colname = self._data.columns.tolist()[column]
        colname='objects'
        self.layoutAboutToBeChanged.emit()
        self._data.sort_values(colname, ascending=order == QtCore.Qt.AscendingOrder, inplace=True)
        self._data.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()




if __name__ == '__main__' :
    # # pow='C:\Users\alibh\Desktop\My_Qt\Paper_2_PyQT_Pytorch\Feature_ext\data\VOCdevkit\VOC2007\Annotations\000001.xml'
    # p=os.path.join(os.getcwd(),'data','VOCdevkit','VOC2007','Annotations','000001.xml')
    # print(p)
    # print(load_annotation(p, {'person':1}))

    cfg_file_path = os.path.join(os.getcwd(), 'config.yml')
    cfg=import_yaml(cfg_file_path)
    path = cfg['dataset']['path']
    data=Read_VOC(cfg)
    # print(data)
    data_pd=pd.DataFrame.from_dict(data)
    print(data_pd)
