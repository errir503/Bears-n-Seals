import io

import Tkinter
from Tkinter import *

import numpy
import numpy as np
from PIL import Image, ImageTk
import os
import glob
import imgaug as ia

# colors for the bounding boxes
from arcticapi import ArcticApi, HotSpot
from arcticapi.augmnetation.AugRgb import shift_boxes, getRectFromYolo, shift_box
from arcticapi.augmnetation.TrainingChip import TrainingChip
from arcticapi.config import load_config

COLORS = ['#a661b6','#3bb218','#e6ee7f']
CLASSES = ["Ringed", "Bearded", "UNK"]
LABELS_DIR = "src/bbox-labeler/relabel-backup"
# LABELS_DIR = "Images"
# csv_in = '/Users/yuval/Documents/XNOR/Bears-n-Seals/src/bbox-labeler/out.csv'
csv_in = '/Users/yuval/Documents/XNOR/Bears-n-Seals/updated_live.csv'
csv_out = '/Users/yuval/Documents/XNOR/Bears-n-Seals/updated_live_out.csv'

# noinspection PyUnusedLocal
class LabelTool(Tkinter.Frame):
    def __init__(self, master):
        Tkinter.Frame.__init__(self, master)
        self.pack()
        self.api = ArcticApi(csv_in, '')
        self.cfg = load_config("labeler_cfg")

        # set up the main frame
        self.parent = master
        self.parent.title("Seal LabelTool")
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
        self.classCandidateFileName = 'src/bbox-labeler/class.txt'

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
            with open(self.classCandidateFileName, "r") as cf:
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


        self.statusBtns = Frame(self.frame)
        self.statusBtns.grid(row=5, column=2, columnspan=3, sticky=W + E)
        self.deleted = Button(self.statusBtns, text='Delete', width=15, command=self.prevImage)
        self.deleted.grid(row=1, column=1, sticky=W + N)
        self.badres = Button(self.statusBtns, text='BadRes', width=15, command=self.prevImage)
        self.badres.grid(row=1, column=2, sticky=W + E + N)
        self.dup = Button(self.statusBtns, text='Duplicate', width=15, command=self.prevImage)
        self.dup.grid(row=1, column=3, sticky=E + N)
        self.croppedme = Button(self.statusBtns, text='CroppedByMe', width=15, command=self.prevImage)
        self.croppedme.grid(row=2, column=1, sticky=W + N)
        self.croppededge = Button(self.statusBtns, text='CroppedOnEdge', width=15, command=self.prevImage)
        self.croppededge.grid(row=2, column=2, sticky=W + E + N)
        self.new = Button(self.statusBtns, text='New', width=15, command=self.prevImage)
        self.new.grid(row=2, column=3, sticky=E + N)

        self.btnDel = Button(self.frame, text='Delete Bbox', command=self.delBBox)
        self.btnDel.grid(row=6, column=2, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=7, column=2, sticky=W + E + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=7, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to img idx.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)
        self.tmpLabel = Label(self.ctrPanel, text="Go to hsID.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idEntry = Entry(self.ctrPanel, width=5)
        self.idEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImageId)
        self.goBtn.pack(side=LEFT)

        self.imglbl = Label(self.ctrPanel, text='Image:')
        self.imglbl.pack(side=LEFT)


        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

        self.imgElements = []

    def loadDir(self, dbg=False):
        # get image list
        self.imageDir = os.path.join(r'./'+LABELS_DIR)
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))
        if len(self.imageList) == 0:
            print('No .JPG images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        self.loadImage()
        print('%d images loaded from %s' % (self.total, './'+LABELS_DIR))

    def rint(self, num):
        return np.int32(np.rint(num))

    def loadImage(self):
        # load image
        imagePath = self.imageList[self.cur - 1]
        self.img = Image.open(imagePath)
        self.img = self.img.resize([int(self.zoom * s) for s in self.img.size], Image.ANTIALIAS)
        self.tkImg = ImageTk.PhotoImage(self.img)
        w = self.tkImg.width()
        h = self.tkImg.height()
        self.mainPanel.config(width=max(self.tkImg.width(), 400), height=max(self.tkImg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        #del old ui elements
        for el in self.imgElements:
            self.mainPanel.delete(el)
        self.imgElements = []

        # load labels
        self.clearBBox()
        self.imageName = os.path.split(imagePath)[-1].split('.jpg')[0]
        labelname = self.imageName + '.2label'
        self.labelFileName = os.path.join(self.outDir, labelname)
        self.labelFileName = imagePath.replace('.' + imagePath.split(".")[-1], ".2label")
        print(self.labelFileName)
        self.imglbl.config(text='Image: ' + self.labelFileName)

        # create the chip object for each frame and update bounding boxes
        npim = numpy.array(self.img)
        aereal_image = None
        hotspots = []
        tcrop, bcrop, lcrop, rcrop = (0,0,0,0)
        if os.path.exists(self.labelFileName):
            with open(self.labelFileName, 'r') as f:
                for (i, line) in enumerate(f):
                    bbox = self.line_to_bbox(line)
                    hs = self.api.hsm.get_hs(bbox.hsId)
                    x1, y1, x2, y2 = getRectFromYolo(npim, bbox.x, bbox.y, bbox.w, bbox.h)
                    hs.update_bbox(x1, y2, x2, y1)
                    hs.rgb_bb = shift_box(hs.rgb_bb, -bbox.lcrop, -bbox.tcrop)
                    if aereal_image is None:
                        aereal_image = self.api.rgb_images[hs.rgb.path]
                    hotspots.append(hs)
                    tcrop,bcrop,lcrop,rcrop=(bbox.tcrop, bbox.bcrop, bbox.lcrop, bbox.rcrop)

        bboxes_in_chip = [hs.rgb_bb for hs in hotspots]
        bboxes_in_chip = shift_boxes(bboxes_in_chip, lcrop, tcrop)
        chip = TrainingChip(aereal_image, npim.shape, self.cfg, bboxes_in_chip, (tcrop, bcrop, lcrop, rcrop ))
        chip.image = npim
        chip.filename = os.path.dirname(chip.filename) + "/" + self.imageName
        self.chip = chip

        for bbs in chip.bboxes.bounding_boxes:
            tmpId = self.mainPanel.create_rectangle(self.rint(bbs.x1),
                                                    self.rint(bbs.y1),
                                                    self.rint(bbs.x2),
                                                    self.rint(bbs.y2),
                                                    width=2,
                                                    outline=COLORS[bbs.label])

            self.bboxIdList.append(tmpId)
            self.bboxList.append(bbs)
            self.listbox.insert(END, self.bbox_string(bbs))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[int(bbs.label)])
            self.globalhsid = bbs.hsId

        # paint stats
        a1 = self.mainPanel.create_text(3, 3, text=("(%d,%d)" % (self.chip.crops[2], self.chip.crops[0])),
                                        anchor="nw", fill="red",font="Times 18 bold")
        a2 = self.mainPanel.create_text(w-3, h-3, text=("(%d,%d)" % (self.chip.crops[3], self.chip.crops[1])),
                                        anchor="se", fill="red",font="Times 18 bold")
        self.imgElements.append(a1)
        self.imgElements.append(a2)

        if len(self.bboxList) > 0:
            self.listbox.selection_set(0)
        else:
            self.globalhsid = 0

    def saveImage(self):
        open(self.chip.filename + ".2label", 'w').close()
        open(self.chip.filename + ".txt", 'w').close()
        self.chip.save()
        # return
        # with io.open(self.labelFileName, 'w', encoding="utf-8") as f:
        #     for bbox in self.bboxList:
        #         newstr = ('%s %s %s %s %s %s %s %s %s %s' % (bbox.hsid, str(bbox.classid), str(bbox.x), str(bbox.y),
        #                   str(bbox.w), str(bbox.h), str(bbox.tcrop), str(bbox.bcrop),
        #                   str(bbox.lcrop), str(bbox.rcrop)))
        #         newstr = unicode(newstr)
        #         f.write(newstr + '\n')

        header = "hotspot_id,timestamp,filt_thermal16,filt_thermal8,filt_color,x_pos,y_pos,thumb_left,thumb_top,thumb_right," \
                 "thumb_bottom,hotspot_type,species_id,updated_bot,updated_top,updated_left,updated_right,updated,status"
        self.api.saveHotspotsToCSV(csv_out, "")

        # with io.open(os.path.splitext(self.labelFileName)[0]+'.txt', 'w', encoding="utf-8") as f:
        #     for bbox in self.bboxList:
        #         newstr = ('%s %s %s %s %s' % (str(bbox.classid), str(bbox.x), str(bbox.y),
        #                   str(bbox.w), str(bbox.h)))
        #         newstr= unicode(newstr)
        #         f.write(newstr + '\n')
        # print('Image No. %d saved' % self.cur)

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
                self.update_bbox(selected_idx[0], x1, y1, x2, y2)
            if len(selected_idx) > 0:
                new_idx = selected_idx[0] + 1
                if not len(self.bboxList) <= new_idx:
                    self.listbox.selection_set(new_idx)
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
        bbox.classid = CLASSES.index(str(self.currentLabelClass))
        bbox.bcrop = self.globalcrops.bcrop
        bbox.tcrop = self.globalcrops.tcrop
        bbox.lcrop = self.globalcrops.lcrop
        bbox.rcrop = self.globalcrops.rcrop
        self.globalhsid = str(round(float(self.globalhsid) + 0.1,1))
        hs = self.api.hsm.get_hs(bbox.globalhsid)
        new_hs = HotSpot()

        bbox.hsId = self.globalhsid
        self.bboxList.append(bbox)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.listbox.insert(END, self.bbox_string(bbox))
        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[int(bbox.classid)])

    def update_bbox(self, bbox_idx, x1, y1, x2, y2):
        w = (abs(x1 - x2) + 0.0) / self.img.size[0]
        h = (abs(y1 - y2) + 0.0) / self.img.size[1]
        xCenter = ((x1 + x2 + 0.0) / 2) / self.img.size[0]
        yCenter = ((y1 + y2 + 0.0) / 2) / self.img.size[1]
        bbox = self.bboxList[bbox_idx]
        hs = self.api.hsm.get_hs(bbox.hsId)
        bbox.x = xCenter
        bbox.y = yCenter
        bbox.w = w
        bbox.h = h
        bbox.classid = hs.rgb_bb.label
        bbox.hsId = hs.rgb_bb.hsId

        x1, y1, x2, y2 = getRectFromYolo(self.chip.image, xCenter, yCenter, w, h)
        hs.update_bbox(x1, y1, x2, y2)
        hs.rgb_bb = shift_box(hs.rgb_bb, -self.chip.crops[2], -self.chip.crops[0])
        hs.updated = True

        updated_boxes = []
        updated_boxes.append(hs.rgb_bb)
        for idx, bbox in enumerate(self.chip.bboxes.bounding_boxes):
            if bbox.hsId != hs.id:
                updated_boxes.append(bbox)
        self.chip.bboxes = ia.BoundingBoxesOnImage(updated_boxes, shape=self.chip.bboxes.shape)


        self.bboxList[bbox_idx] = bbox
        self.mainPanel.delete(self.bboxIdList[bbox_idx])
        self.bboxIdList[bbox_idx] = (self.bboxId)
        self.bboxId = None
        self.listbox.delete(0, END)

        for bbox in self.bboxList:
            self.listbox.insert(END, self.bbox_string(bbox))

        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[hs.rgb_bb.label])

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

        hs = self.api.hsm.get_hs(self.bboxList[idx].hsId)
        hs.status = "removed"

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

    def gotoImageId(self):
        id = str(self.idEntry.get())
        idx = None
        for imidx, imgpath in enumerate(self.imageList):
            if id in imgpath:
                idx = imidx
                break

        if 1 <= idx <= self.total and idx is not None:
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
        bbox.hsId = tmp[0]
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
        hs = self.api.hsm.get_hs(bbox.hsId)
        other_hs_in_im = [a for a in self.api.rgb_images[hs.rgb.path].hotspots if a.id != hs.id]

        # l = hs.updated_left-bbox.lcrop
        # r = hs.updated_right - bbox.lcrop
        # t = hs.updated_top - bbox.tcrop
        # b = hs.updated_bot -bbox.tcrop
        # a1 = self.mainPanel.create_text(l, b, text="status: %s updated: %s" % (hs.status, str(hs.updated)), anchor="nw",
        #                                 fill="green")
        # a2 = self.mainPanel.create_text(l, t, text=hs.id, anchor="nw", fill="red")
        if hs.updated:
            self.drawUiBoxForUpdated(hs, self.chip.crops[2], self.chip.crops[0], "red")
        for otherhs in other_hs_in_im:
            self.drawUiBoxForUpdated(otherhs, self.chip.crops[2], self.chip.crops[0], "#000")


        b, t, l, r = hs.getBTLR(True)
        l_orig = l - self.chip.crops[2]
        r_orig = r - self.chip.crops[2]
        t_orig = t - self.chip.crops[0]
        b_orig = b - self.chip.crops[0]
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_y = b_orig + ((t_orig - b_orig) / 2)
        box2 = self.mainPanel.create_oval(center_x-3 , center_y-3,center_x+3 , center_y+3, width=0, fill='white', outline="#FFF")
        self.imgElements.append(box2)

        updated_str = "U" if hs != None and hs.updated else "NU"
        return ('%s %s : %s (b:%d, t:%d) (l:%d, r:%d)' % (updated_str, bbox.hsId, CLASSES[bbox.label], b_orig,
                                                          t_orig, l_orig, r_orig))

    def drawUiBoxForUpdated(self, hs, lcrop, tcrop, color):
        l = hs.updated_left - lcrop
        r = hs.updated_right - lcrop
        t = hs.updated_top - tcrop
        b = hs.updated_bot - tcrop
        box1 = self.mainPanel.create_rectangle(l, b, r, t, width=2, outline=color)
        a1 = self.mainPanel.create_text(l, b, text="status: %s updated: %s" % (hs.status, str(hs.updated)), anchor="nw",
                                        fill="green")
        a2 = self.mainPanel.create_text(l, t, text=hs.id, anchor="nw", fill="red")

        self.imgElements.append(a1)
        self.imgElements.append(a2)
        self.imgElements.append(box1)


class Object(object):
    pass

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    tool.loadDir()
    root.resizable(width=True, height=True)
    root.update()
    root.mainloop()
