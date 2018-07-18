import numpy as np
from glob import glob
import cv2
from pyprind import prog_percent
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
import re
import pyqtgraph as pg
from skimage.feature import match_template

def rgba_to_RGB(*args):
    '''Converts rgba values in float to RGB values in uint8'''
    return tuple([int(i*255) for i in args[:-1]])

def findEyes(im):
    t = (im > 15) & (im < 25)
    im2, contours, hierarchy = cv2.findContours((t * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours

def getBoundingBoxCenter(c):
    br = cv2.boundingRect(
        np.array([c[0].max(0).flatten(), c[1].max(0).flatten(), c[0].min(0).flatten(), c[1].min(0).flatten()]))
    center = br[0] + br[2] // 2, br[1] + br[3] // 2
    return center

def findEyeLocationTM(im, eyes_template):
    '''
    Returns peak of probable eye location using 2D template matching
    :param im: The image file
    :param eyes_template: the eye template as a 2D numpy array
    :return: the center of head position as tuple
    '''
    m = match_template(im, eyes_template)
    return np.unravel_index(np.argmax(m), m.shape)[::-1]

class Log:
    def __init__(self):
        '''
            parses log files to look for annotated tail tip and base.
        '''
        self.tailtip = re.compile(r'Position "tailtip" was set to \(([0-9]{2,}),([0-9]{2,})\)!')
        self.tailbase = re.compile(r'Position "tailbase" was set to \(([0-9]{2,}),([0-9]{2,})\)!')

    def getTail(self, fn):
        '''
        Reads log file and returns tail tip and base coordinates.
        :param fn:
        :return:
        '''
        with open(fn) as fp:
            # Read log file entries
            log = ''.join([i.strip() for i in fp.readlines()])

            # Find tail base and tip "clicks", use the last one (as it was used for the experiment)
            tb = np.array([int(i) for i in self.tailbase.findall(log)[-1]])
            tt = np.array([int(i) for i in self.tailtip.findall(log)[-1]])

        return tb, tt


class Annotator(QWidget):
    def __init__(self, folder=None):
        '''
            Initializes Annotator
        '''
        super().__init__()
        self.offset_x = 10
        self.offset_y = 10

        self.colors = [rgba_to_RGB(0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
                       rgba_to_RGB(1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
                       rgba_to_RGB(0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
                       rgba_to_RGB(0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0)]

        if folder is None:
            self.folder = r'C:\Users\me\Documents\MPIN\fish_images'
            self.folder_logs = r'C:\Users\me\Documents\MPIN\fish_images\logs'

        self.eyes_template = cv2.imread(r"C:\Users\me\Documents\MPIN\fish_images\eye_template\eye_template.png", 0)

        self.files = sorted(glob(self.folder+'\\*.png'))
        print(self.files, 'files found.')

        self.stack = self.loadfiles()
        self.stack_rgb = np.repeat(self.stack[...,None], 3, 3)

        self.processStack()

        self.imviewer = ImageViewer(np.transpose(self.stack_rgb, (0, 2, 1, 3)))
        self.l = QGridLayout()
        self.l.addWidget(self.imviewer)

        self.setLayout(self.l)

    def loadfiles(self):
        stack = []

        for fn in prog_percent(self.files):
            im = cv2.imread(fn, 0).astype(np.float32)
            if im.shape == (488,648):
                im = im[4:-4, 4:-4]

            stack.append(im)

        return np.array(stack, dtype=np.float32)

    def processStack(self):
        L = Log()
        r = 40

        for i, im in enumerate(self.stack_rgb):
            im_to_save = im[...,0].copy()
            # c = findEyes(im)
            # center = getBoundingBoxCenter(c)

            center = findEyeLocationTM(im_to_save, self.eyes_template)

            try:
                # Get filename
                fn = self.files[i].split('\\')[-1]
                # Get tail base and tip coordinates from log file
                tb, tt = L.getTail(self.folder_logs+'\\'+fn.replace('image.png', 'log.txt'))

                # Draw rectangles for head, tail tip and base, and background to indicate where the coordinates are
                cv2.rectangle(im, (center[0], center[1]), (center[0] + 2*r, center[1] + 2*r), self.colors[0], 4)
                cv2.rectangle(im, tuple(tb - r), tuple(tb + r),  self.colors[2], 4)
                cv2.rectangle(im, tuple(tt - r), tuple(tt + r),  self.colors[1], 4)
                cv2.rectangle(im, (self.offset_x, self.offset_y), (self.offset_x+r*2, self.offset_y+r*2),
                              self.colors[3], 4)

#                cv2.imwrite(self.folder + '\\head\\' + fn, im_to_save[center[1] - r:center[1] + r, center[0] - r:center[0] + r])

                # Write image regions to files
                cv2.imwrite(self.folder + '\\head\\' + fn,
                            im_to_save[center[1]:center[1] + 2*r, center[0]:center[0]+2*r])
                cv2.imwrite(self.folder + '\\tailtip\\' + fn, im_to_save[tt[1] - r:tt[1] + r, tt[0] - r:tt[0] + r])
                cv2.imwrite(self.folder + '\\tailbase\\' + fn, im_to_save[tb[1] - r:tb[1] + r, tb[0] - r:tb[0] + r])
                cv2.imwrite(self.folder + '\\background\\' + fn, im_to_save[self.offset_x:2*r+self.offset_x,
                                                                 self.offset_y:2*r+self.offset_y])

            except:
                print('log file not found...')


class ImageViewer(pg.ImageView):
    def __init__(self, stack):
        '''
        Helper class for showing a 3D stack inside of widget using pyqtgraph
        :param stack: 3D stack of files x X x Y
        '''
        super().__init__()
        self.stack = stack
        self.setImage(self.stack)


if __name__ == '__main__':
    app = QApplication([])
    a = Annotator()
    a.show()

    app.exec_()