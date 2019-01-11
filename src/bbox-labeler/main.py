import tkinter
from tkinter import *

import numpy as np
from PIL import Image, ImageTk
import os
import glob

# colors for the bounding boxes
COLORS = ['#a661b6','#3bb218','#e6ee7f']
CLASSES = ["Ringed", "Bearded", "UNK"]

# noinspection PyUnusedLocal
class LabelTool(tkinter.Frame):
    def __init__(self, master):
        tkinter.Frame.__init__(self, master)
        self.pack()
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=True, height=True)
        # initialize global state
        self.imageDir = ''
        self.imageList = []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = None
        self.imageName = ''
        self.img = None
        self.labelFileName = ''
        self.tkImg = None
        self.currentLabelClass = ''
        self.cla_can_temp = []
        self.classCandidateFileName = 'class.txt'

        # initialize mouse state
        self.STATE = dict()
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # prep for zoom
        self.zoom = 1

        self.globalhsid = None
        self.globalcrops = Object()

        # ----------------- GUI stuff ---------------------
        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("c", self.clearBBox)  # press c to clear bbox
        self.parent.bind("u", lambda e: self.listbox.selection_clear(0, END))  # press u to unselect all in listbox
        self.parent.bind("<Left>", self.prevImage)  # press 'a' to go backward
        self.parent.bind("<Right>", self.nextImage)  # press 'd' to go forward
        self.parent.bind("<space>", self.nextImage)  # press space to go forward
        self.parent.bind("<Down>", self.zoom_out)  # press down arrow to zoom out
        self.parent.bind("<Up>", self.zoom_in)  # press up arrow to zoom in
        self.parent.bind("<Delete>", self.delBBox_key)  # press 'delete' to delete selected box
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # choose class
        if os.path.exists(self.classCandidateFileName):
            with open(self.classCandidateFileName, encoding='utf-8-sig') as cf:
                for line in cf.readlines():
                    tmp = line.strip('\n')
                    self.cla_can_temp.append(tmp)

        self.className = StringVar()
        self.className.set(self.cla_can_temp[0])
        self.classCandidate = OptionMenu(self.frame, self.className, *self.cla_can_temp, command=self.setClass)
        self.currentLabelClass = self.className.get()  # init
        self.classCandidate.grid(row=1, column=2)

        self.btnClass = Button(self.frame, text='ComfirmClass', command=self.setClass)
        self.btnClass.grid(row=2, column=2, sticky=W + E)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text='Bounding boxes:')
        self.lb1.grid(row=3, column=2, sticky=W + N)
        self.listbox = Listbox(self.frame, width=44, height=12)
        self.listbox.grid(row=4, column=2, sticky=N + S)
        self.btnDel = Button(self.frame, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=5, column=2, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=6, column=2, sticky=W + E + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=7, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        self.imglbl = Label(self.ctrPanel, text='Image:')
        self.imglbl.pack(side=LEFT)


        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)


        # CSV
        # self.csvlines = {}
        # idx = 1
        # rows = list()
        # f = open("_CHESS_ImagesSelected4Detection.csv", 'r')
        # reader = csv.reader(f)
        # for row in reader:
        #     rows.append(row)
        # f.close()
        # del rows[0]  # remove col headers
        # for row in rows:
        #     row.append(idx)
        #     self.csvlines[row[0]] = row
        #     idx+=1
        #
        #
    def loadDir(self, dbg=False):
        # get image list
        self.imageDir = os.path.join(r'./Images')
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'), recursive=True)
        if len(self.imageList) == 0:
            print('No .JPG images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # set up output dir
        # self.outDir = os.path.join(r'./Labels', '%s' % self.category)
        # if not os.path.exists(self.outDir):
        #     os.mkdir(self.outDir)

        self.loadImage()
        print('%d images loaded from %s' % (self.total, './Images'))

    def rint(self, num):
        return np.int32(np.rint(num))

    def loadImage(self):
        # load image
        imagePath = self.imageList[self.cur - 1]
        self.img = Image.open(imagePath)
        self.img = self.img.resize([int(self.zoom * s) for s in self.img.size], Image.ANTIALIAS)
        self.tkImg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkImg.width(), 400), height=max(self.tkImg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imageName = os.path.split(imagePath)[-1].split('.')[0]
        labelname = self.imageName + '.2label'
        self.labelFileName = os.path.join(self.outDir, labelname)
        self.labelFileName = imagePath.replace('.' + imagePath.split(".")[-1], ".2label")
        print(self.labelFileName)
        self.imglbl.config(text='Image: ' + self.labelFileName)
        if os.path.exists(self.labelFileName):
            with open(self.labelFileName, encoding="utf-8") as f:
                for (i, line) in enumerate(f):

                    bbox = self.line_to_bbox(line)

                    w = self.tkImg.width()
                    h = self.tkImg.height()
                    box_w = self.rint(float(bbox.w * w) * self.zoom)
                    box_h = float(bbox.h * h) * self.zoom
                    center_x = self.rint(bbox.x * self.zoom  * w)
                    center_y = self.rint(bbox.y * self.zoom  * h)

                    tmpId = self.mainPanel.create_rectangle(self.rint(center_x - box_w/2),
                                                            self.rint(center_y - box_h/2),
                                                            self.rint(center_x + box_w/2),
                                                            self.rint(center_y + box_h/2),
                                                            width=2,
                                                            outline=COLORS[bbox.classid])

                    self.bboxIdList.append(tmpId)
                    self.bboxList.append(bbox)
                    self.listbox.insert(END, self.bbox_string(bbox))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[int(bbox.classid)])
                    self.globalhsid = bbox.hsid
                    self.globalcrops.bcrop = bbox.bcrop
                    self.globalcrops.tcrop = bbox.tcrop
                    self.globalcrops.lcrop = bbox.lcrop
                    self.globalcrops.rcrop = bbox.rcrop
        if len(self.bboxList) > 0:
            self.listbox.selection_set(0)
        else:
            self.globalhsid = 0

    def saveImage(self):
        with open(self.labelFileName, 'w', encoding="utf-8") as f:
            for bbox in self.bboxList:
                newstr = ('%s %s %s %s %s %s %s %s %s %s' % (bbox.hsid, str(bbox.classid), str(bbox.x), str(bbox.y),
                          str(bbox.w), str(bbox.h), str(bbox.tcrop), str(bbox.bcrop),
                          str(bbox.lcrop), str(bbox.rcrop)))
                f.write(newstr + '\n')

        with open(os.path.splitext(self.labelFileName)[0]+'.txt', 'w', encoding="utf-8") as f:
            for bbox in self.bboxList:
                newstr = ('%s %s %s %s %s' % (str(bbox.classid), str(bbox.x), str(bbox.y),
                          str(bbox.w), str(bbox.h)))
                f.write(newstr + '\n')
        print('Image No. %d saved' % self.cur)
        pass

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = self.rint(event.x / self.zoom), self.rint(event.y / self.zoom)
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)

            selected_idx = self.listbox.curselection()
            item = None
            if len(selected_idx) > 0:
                item = self.bboxList[selected_idx[0]]

            if item is None:
                self.append_new_bbox(x1, x2, y1, y2)
            else:
                self.update_bbox(selected_idx[0], x1, x2, y1, y2)
        self.STATE['click'] = 1 - self.STATE['click']

    def append_new_bbox(self, x1, x2, y1, y2):
        w = (abs(x1 - x2) + 0.0) / self.img.size[0]
        h = (abs(y1 - y2) + 0.0) / self.img.size[1]
        xCenter = ((x1 + x2 + 0.0) / 2) / self.img.size[0]
        yCenter = ((y1 + y2 + 0.0) / 2) / self.img.size[1]
        bbox = Object()
        bbox.w = w
        bbox.h = h
        bbox.x = xCenter
        bbox.y = yCenter
        bbox.classid = CLASSES.index(self.currentLabelClass)
        bbox.bcrop = self.globalcrops.bcrop
        bbox.tcrop = self.globalcrops.tcrop
        bbox.lcrop = self.globalcrops.lcrop
        bbox.rcrop = self.globalcrops.rcrop
        self.globalhsid = str(round(float(self.globalhsid) + 0.1,1))
        bbox.hsid = self.globalhsid
        self.bboxList.append(bbox)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.listbox.insert(END, self.bbox_string(bbox))
        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[int(bbox.classid)])

    def update_bbox(self, bbox_idx, x1, x2, y1, y2):
        w = (abs(x1 - x2) + 0.0) / self.img.size[0]
        h = (abs(y1 - y2) + 0.0) / self.img.size[1]
        xCenter = ((x1 + x2 + 0.0) / 2) / self.img.size[0]
        yCenter = ((y1 + y2 + 0.0) / 2) / self.img.size[1]
        bbox = self.bboxList[bbox_idx]
        bbox.x = xCenter
        bbox.y = yCenter
        bbox.w = w
        bbox.h = h
        bbox.lcrop = self.globalcrops.lcrop
        bbox.rcrop = self.globalcrops.rcrop
        bbox.tcrop = self.globalcrops.tcrop
        bbox.bcrop = self.globalcrops.bcrop
        self.bboxList[bbox_idx] = bbox
        self.mainPanel.delete(self.bboxIdList[bbox_idx])
        self.bboxIdList[bbox_idx] = (self.bboxId)
        self.bboxId = None
        self.listbox.delete(0, END)
        for bbox in self.bboxList:
            self.listbox.insert(END, self.bbox_string(bbox))
        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[int(bbox.classid)])

    def mouseMove(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkImg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkImg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkImg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.rint(self.STATE['x'] * self.zoom),
                                                          self.rint(self.STATE['y'] * self.zoom),
                                                          self.rint(event.x * self.zoom),
                                                          self.rint(event.y * self.zoom),
                                                        width=2, outline=COLORS[0])
    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1:
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def delBBox_key(self, event=None):
        self.delBBox()

    def clearBBox(self, event=None):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

    def setClass(self, event=None):
        self.currentLabelClass = self.className.get()
        print('set label class to :', self.currentLabelClass)

    def zoom_in(self, event=None):
        self.zoom *= 1.2
        self.saveImage()
        self.loadImage()

    def zoom_out(self, event=None):
        self.zoom /= 1.2
        self.saveImage()
        self.loadImage()

    def line_to_bbox(self, line):
        # Line Format:
        # hsid classid x y w h topcrop bottomcrop leftcrop rightcrop
        tmp2 = [t.strip() for t in line.split()]
        tmp = [float(t) if idx > 1 and idx < 6 else t for idx, t in enumerate(tmp2)]
        tmp[1] = int(tmp[1])
        tmp[6] = int(tmp[6])
        tmp[7] = int(tmp[7])
        tmp[8] = int(tmp[8])
        tmp[9] = int(tmp[9])
        bbox = Object()
        bbox.hsid = tmp[0]
        bbox.classid = tmp[1]
        bbox.x = tmp[2]
        bbox.y = tmp[3]
        bbox.w = tmp[4]
        bbox.h = tmp[5]
        bbox.tcrop = tmp[6]
        bbox.bcrop = tmp[7]
        bbox.lcrop = tmp[8]
        bbox.rcrop = tmp[9]
        return bbox

    def bbox_string(self, bbox):
        return ('%s : %s (x:%.3f, y:%.3f) (w:%.3f, h:%.3f)' % (bbox.hsid, CLASSES[bbox.classid], bbox.x,
                                                  bbox.y, bbox.w, bbox.h))

class Object(object):
    pass

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    tool.loadDir()
    root.resizable(width=True, height=True)
    root.mainloop()
