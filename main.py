import os
os.system("pyuic5 ./main_ui.ui > ./main_ui.py")

import  sys
from main_ui import    Ui_MainWindow
import utils_2 as util
import pandas as pd
import cv2

from PyQt5.QtWidgets import QApplication , QMainWindow
from  PyQt5.QtCore import QModelIndex , QObject ,pyqtSignal, pyqtSlot , QRunnable , QThreadPool
import traceback


def table_update(filter1, data_pd, config):
    filter1 = eval(filter1)
    if (config['filter_only_contain']):
        k = data_pd.apply(lambda x: all([k in (x.cls) for k in filter1]), axis=1)
    else:
        k = data_pd.apply(lambda x: any([k in (x.cls) for k in filter1]), axis=1)
    data_pd = pd.DataFrame.from_dict(data_pd[k])
    # print(data_pd[k].index)


    return data_pd




def thread_complete():
    print("Thread complete.")

def progress_fn():
    print("In the thread.")


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        No data
    error
        `tuple` (exctype, value, traceback.format_exc() )
    result
        `object` data returned from processing, anything
    progress
        `int` indicating % progress
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class Appwin(QMainWindow):
    def __init__(self):
        super(Appwin,self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.setMinimumWidth(resolution.width()/3.0)
        self.setMinimumHeight(resolution.height()/1.5)
        self.config = util.import_yaml(os.path.join(os.getcwd(),'config','config.yml'))


        root = self.config['dataset']['path']
        print(util.check_VOC(root))
        self.data, self.category_index = util.Read_VOC(self.config)
        # print(self.data)
        self.config_update()
        self.ui.tableView.clicked.connect(self.table_clicked)
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.ui.btn_filter.clicked.connect(self.btn_filter)
        self.ui.btn_gen_text.clicked.connect(self.btn_gen_text)
        self.ui.btn_create_dataloader_torch.clicked.connect(self.btn_torch_data_loader)


    def config_update(self):



        data_pd = pd.DataFrame.from_dict(self.data)
        data_pd['objects']=data_pd.apply(lambda row: len(row.cls),axis=1)
        col_name = list(data_pd.columns)
        data_pd['area'] = data_pd.apply(
            lambda x: ((x[col_name[3]] - x[col_name[4]])[:, 1] * (x[col_name[3]] - x[col_name[4]])[:, 0]), axis=1)

        data_pd['center_point'] = data_pd.apply(
            lambda x: ((x[col_name[4]])+(x[col_name[3]]))/2.0, axis=1)

        data_pd['area'] = data_pd.apply(util.calculate_area, axis=1)

        print("[Read image sizes]... it takes several Minutes...")
        data_pd['image_size'] = data_pd.apply(util.get_image_size, axis=1)

        if (self.ui.chck_box_scale_to_1.isChecked()):
            data_pd['yx_max'] = data_pd.apply(
                lambda x: (x['yx_max'] / x['image_size'] ), axis=1)
            data_pd['yx_min'] = data_pd.apply(
                lambda x: (x['yx_min'] / x['image_size'] ), axis=1)
            data_pd['center_point'] = data_pd.apply(
                lambda x: (x['center_point'] / x['image_size'] ), axis=1)
            data_pd['area'] = data_pd.apply(
                lambda x: ((x[col_name[3]] - x[col_name[4]])[:, 1] * (x[col_name[3]] - x[col_name[4]])[:, 0]), axis=1)


        # print(data_pd.apply(lambda x: x[0]))
        self.data_pd = data_pd
        self.model = util.pandasModel_2(data_pd)
        self.ui.tableView.setModel(self.model)
        # self.ui..addItems(self.cfg['dataset']['category'])
        list_of_cls=[str(i) + ' (' + str(self.category_index[i]) + ')' for i in self.category_index]

        self.ui.comboBox.addItems(list_of_cls)


        self.config['filter_only_contain']= self.ui.radio_btn_filter_only.isChecked()










    def btn_filter(self):
        self.config_update()
        filter1 = self.ui.in_filter_class.text()
        data=self.data_pd
        config=self.config
        # Pass the function to execute
        worker = Worker(table_update ,filter1,data ,config)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(thread_complete)
        worker.signals.progress.connect(progress_fn)


        # Execute
        self.threadpool.start(worker)






        #
        # zz=self.data_pd[10:40]
        # self.model = util.PandasModel(zz)
        # self.ui.tableView.setModel(self.model)
        # print(self.data_pd[k])

    def btn_torch_data_loader(self):
        # with open('.\data.txt','r') as f:
        #     data=f.read()
        self.ui.textBrowser.clear()
        # self.ui.textBrowser.setText(str((data[1:,:])))

        data_pd= pd.read_csv('.\data.txt', sep=" ")
        data_pd=data_pd.groupby('Name').agg(list)
        # self.ui.textBrowser.append(str(data_pd))

        # for i in range((data_pd.shape[0])):
        #     self.ui.textBrowser.append(str(data_pd.index.values[i]))
        #     self.ui.textBrowser.append(str(data_pd.iloc[i].tolist()))

        for i in range((data_pd.shape[0])):
            b = list(data_pd.iloc[i])
            self.ui.textBrowser.append(str(('[{}]'.format(data_pd.index.values[i]))))
            #     print(data_pd1.iloc[i].tolist())
            for obj in range(len(b[0])):
                obj = [row[obj] for row in b]
                self.ui.textBrowser.append(str((''.join(str(obj)).strip('[]').replace(',', ''))))

        with open ('data_new_format.txt','w') as file:
            file.write(str(self.ui.textBrowser.toPlainText()))

    def table_clicked(self,index):
        index=QModelIndex(index)

        # ind.__setattr__('column',0)
        row , col=index.row() ,index.column()
        ind = self.model.index(row, 2)
        image= list(self.model.itemData(QModelIndex(ind)).values())[0]

        ind = self.model.index(row, 3)
        yxMax= list(self.model.itemData(QModelIndex(ind)).values())[0]
        yxMax=yxMax.replace('[', '').replace(']', '').split()
        yxMax = [int(float(i)) for i in yxMax]
        yxMax=[tuple(yxMax[i + 0:2 + i]) for i in range(0, len(yxMax), 2)]


        ind = self.model.index(row, 4)
        yxMin = list(self.model.itemData(QModelIndex(ind)).values())[0]
        yxMin = yxMin.replace('[', '').replace(']', '').split()
        yxMin = [int(float(i)) for i in yxMin ]
        yxMin = [tuple(yxMin[i + 0:2 + i]) for i in range(0, len(yxMin), 2)]

        ind = self.model.index(row, 7)
        center = list(self.model.itemData(QModelIndex(ind)).values())[0]
        center = center.replace('[', '').replace(']', '').split()
        center = [int(float(i)) for i in center ]
        center = [tuple(center[i + 0:2 + i]) for i in range(0, len(center), 2)]
        # print(center)

        classes=self.config['dataset']['category'].split(',')
        ind = self.model.index(row, 0)
        label = list(self.model.itemData(QModelIndex(ind)).values())[0]
        label = label.replace('[', '').replace(']', '').split()

        # yxMin=ast.literal_eval(yxMin.replace(' ', ','))

        classes= self.config['dataset']['category'].split(',')

        name=str(image.split('\\')[-1])
        image=cv2.imread(image)

        objs=[]

        for i in range(len(yxMax)):



            y1, x1 = yxMin[i]
            y2, x2 = yxMax[i]
        #
        #
            # print(i,'value: ',x1, y1 ,x2, y2 )
        #
        #
            start_point= (x1,y1)
            end_point = (x2,y2)
            color = (255, 0, 0)
            thickness = 2
            objs.append("{}-{}".format(i+1,classes[int(label[i])]))
            cv2.putText(image, '{}-{}({})'.format(i+1,classes[int(label[i])],label[i]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255),1)
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image=cv2.circle(image, (center[i][1],center[i][0]), 2, (0, 0, 255), -1)

        # util.Qlogging(self.ui.textBrowser, '{} has {} objects: \\n {}'.format(name,len(yxMax),objs),'red' )

        cv2.imshow(name, image)
        cv2.waitKey(4000)
        cv2.destroyAllWindows()
        # print(index.row() ,list(self.model.itemData(QModelIndex(ind)).values())[0])



        # util.Qlogging(self.ui.textBrowser,'image: {},{}'.format(index.data()))





        # self.ui.textBrowser

    def print_output(self, data_pd):
        print("Resault.")
        self.data_pd=data_pd
        self.model = util.pandasModel_2(data_pd)
        self.ui.tableView.setModel(self.model)
        self.ui.lbl_num_objects.setText('Number of Objects: {}'.format(data_pd.shape[0]))

    def btn_gen_text(self):
        self.ui.textBrowser.clear()
        sstr = self.ui.in_text_format.text()
        sstr=" ".join(sstr.split())
        s = False
        e = False

        newStr = ''

        for i in range(len(sstr)):
            if (sstr[i] == ':'):
                s = True

            if (sstr[i] == '}'):
                s = False

            if (s):
                pass
            #         del st[i]
            #         print(st[i])
            else:
                newStr = newStr + sstr[i]
        # sstr=sstr.replace('0','')
        print(newStr)

        sstr = newStr.format( 'class' , 'yMax','xMax', 'yMin','xMin', 'H' ,'W','xCenter','yCenter','Diff','Name')
        self.ui.textBrowser.append(sstr)

        for i in range((self.data_pd.shape[0])):
            print((self.data_pd.shape[0]))
            l=list(self.data_pd.iloc[i,[0,1,2,3,4,6,7]])
            cls=l[0]
            diff = l[1]
            path = l[2]
            path=path.split('\\')[-1]
            yx_min = l[3]
            yx_max = l[4]
            HW   = l[5]
            center = l[6]





            for i in range(len(cls)):
                sstr=self.ui.in_text_format.text()
                sstr = " ".join(sstr.split())
                # sstr='{:02} {:03} {:03} {:03} {:03} {:03} {:03} {:03} {:03} {} {}'.format(cls[i] ,
                sstr = sstr.format(cls[i],
                                             int(yx_min[i,0]) ,int(yx_min[i,1])  ,
                                             int(yx_max[i,0]) ,int(yx_max[i,1]) ,
                                             int(HW[i,0]) , int(HW[i,1]),
                                             int(center[i,0]), int(center[i,1]),
                                              diff[i], path           )
                self.ui.textBrowser.append(sstr)

        print('self.ui.textBrowser.document()')

        with open ('data.txt','w') as file:
            file.write(str(self.ui.textBrowser.toPlainText()))








if __name__ == '__main__':
    app= QApplication(sys.argv)
    desktop= QApplication.desktop()
    resolution = desktop.availableGeometry()
    window = Appwin()

    window.setWindowOpacity(.95)
    window.show()
    window.move(resolution.center()-window.rect().center())
    sys.exit(app.exec())
