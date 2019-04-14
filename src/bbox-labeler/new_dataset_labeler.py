import Tkinter
import cv2
from Tkinter import *

import numpy as np
from PIL import Image, ImageTk
import os
import imgaug as ia
from src.arcticapi.augmnetation.AugRgb import shift_box
from src.arcticapi.config import load_config

from src.arcticapi import ArcticApi, HotSpot
from src.arcticapi.model.HotSpot import SpeciesList

# colors for the bounding boxes
# COLORS = ['#a661b6','#3bb218','#e6ee7f']
COLORS = ['#a661b6','#3bb218','#00FFFF','#000000','#008080', '#800000', '#585656', '#a7a9a9']
CLASSES = ["Ringed", "Bearded", "UNK"]
csv_out = '/fast/fps/fps.csv'


# noinspection PyUnusedLocal
class LabelTool(Tkinter.Frame):

    def __init__(self, master):
        Tkinter.Frame.__init__(self, master)
        self.pack()
        self.cfg = load_config("fps")
        self.api = ArcticApi(self.cfg)
        self.image_names = self.api.getImagesWithSeals()
        self.image_idx = 0
        self.chips = []
        self.chip_idx = 0
        self.m = 0

        # set up the main frame
        self.parent = master
        self.parent.title("Seal LabelTool")
        self.novi = Toplevel()
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=True, height=True)
        # initialize global state
        self.imageDir = ''
        self.imageList = []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.image_idx = 0
        self.total = len(self.image_names)
        self.category = None
        self.imageName = ''
        self.img = None
        self.labelFileName = ''
        self.tkImg = None
        self.currentLabelClass = ''
        self.cla_can_temp = []


        # initialize mouse state
        self.STATE = dict()
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0
        self.SHOW_UI = True

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None

        # prep for zoom
        self.zoom = 1

        self.globalhsid = None

        # ----------------- GUI stuff ---------------------
        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("c", self.clearBBox)  # press c to clear bbox
        self.parent.bind("t", lambda e: self.toggleUI())  # press t to toggle ui labels
        self.parent.bind("u", lambda e: self.listbox.selection_clear(0, END))  # press u to unselect all in listbox
        self.parent.bind("<Left>", self.prevChip)  # press 'a' to go backward
        self.parent.bind("<Right>", self.nextChip)  # press 'd' to go forward
        self.parent.bind("<space>", self.nextChip)  # press space to go forward
        # self.parent.bind("<Down>", self.zoom_out)  # press down arrow to zoom out
        # self.parent.bind("<Up>", self.zoom_in)  # press up arrow to zoom in
        self.parent.bind("<Delete>", self.delBBox_key)  # press 'delete' to delete selected box
        self.parent.bind('<Shift-Right>', lambda e: self.moveChip("right"))
        self.parent.bind('<Shift-Left>', lambda e: self.moveChip("left"))
        self.parent.bind('<Shift-Up>', lambda e: self.moveChip("up"))
        self.parent.bind('<Shift-Down>', lambda e: self.moveChip("down"))


        self.mainPanel.grid(row=1, column=1, rowspan=5, sticky=W + N)

        self.cla_can_temp = SpeciesList

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
        self.off_edge = Button(self.statusBtns, text='OffEdge', width=15, command=lambda: self.set_status('off_edge'))
        self.off_edge.grid(row=1, column=1, sticky=W + N)
        self.badres = Button(self.statusBtns, text='BadRes', width=15, command=lambda: self.set_status('bad_res'))
        self.badres.grid(row=1, column=2, sticky=W + E + N)
        self.dup = Button(self.statusBtns, text='None', width=15, command=lambda: self.set_status('none'))
        self.dup.grid(row=1, column=3, sticky=E + N)
        self.toggleUiElems = Button(self.statusBtns, text='ToggleUI', width=15, command=self.toggleUI)
        self.toggleUiElems.grid(row=2, column=1, sticky=E + N)
        self.toggleUiElems = Button(self.statusBtns, text='Maybe', width=15, command=lambda: self.set_status('maybe_seal'))
        self.toggleUiElems.grid(row=2, column=2, sticky=E + N)
        self.toggleUiElems = Button(self.statusBtns, text='Duplicate', width=15, command=lambda: self.set_type('Duplicate'))
        self.toggleUiElems.grid(row=3, column=1, sticky=E + N)
        self.toggleUiElems = Button(self.statusBtns, text='Animal', width=15,
                                    command=lambda: self.set_type('Animal'))
        self.toggleUiElems.grid(row=3, column=2, sticky=E + N)
        self.toggleUiElems = Button(self.statusBtns, text='IR', width=15, command=self.openIr)
        self.toggleUiElems.grid(row=3, column=3, sticky=E + N)

        self.btnDel = Button(self.frame, text='Delete Bbox', command=self.delBBox)
        self.btnDel.grid(row=6, column=2, sticky=W + E + N)
        self.btnClear = Button(self.frame, text='ClearAll', command=self.clearBBox)
        self.btnClear.grid(row=7, column=2, sticky=W + E + N)



        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=7, column=1, columnspan=2, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevChip)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextChip)
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

        self.loadImage()

    def moveChip(self, direction):
        crops = [x for x in self.chip.crops]
        #(topcrop, bottomcrop, leftcrop, rightcrop)
        stride = 10
        updated = False
        if direction == "up" and crops[0] >= stride:
            crops[0] -= stride
            crops[1] -= stride
            updated = True
            self.chip.shift(0, stride)
        elif direction == "down" and crops[1] <= self.chip.aeral_image.h - stride:
            crops[0] += stride
            crops[1] += stride
            updated = True
            self.chip.shift(0, -stride)
        elif direction == "left" and crops[2] > stride:
            crops[2] -= stride
            crops[3] -= stride
            self.chip.shift(stride, 0)
            updated = True
        elif direction == "right" and crops[3] <= self.chip.aeral_image.w - stride:
            crops[2] += stride
            crops[3] += stride
            self.chip.shift(-stride, 0)
            updated = True
        if not updated:
            return

        self.chip.crops = (crops[0], crops[1], crops[2], crops[3])
        self.loadImage(chip = self.chip)


    def rint(self, num):
        return np.int32(np.rint(num))

    def getCurrentImage(self):
        if self.image_idx >= len(self.image_names):
            print("Requested image index out of range")
            return None
        imagePath = self.image_names[self.image_idx - 1]

        aereal_image = self.api.rgb_images[imagePath]
        if aereal_image is None:
            print("Image path does not exist in aeral images")
            return None
        for chip in self.chips:
            chip.free()
        self.chips = aereal_image.generate_all_chips(self.cfg)
        return aereal_image

    def loadImage(self, refresh=False, chip = None):
        aereal_image = self.getCurrentImage()
        if chip is None:
            self.chip = self.chips[self.chip_idx]
        else:
            self.chip = chip

        if not refresh:
            self.chip.load()
            self.img = ImageTk.PhotoImage(image=Image.fromarray(self.chip.image))
            self.tkImg = ImageTk.PhotoImage(image=Image.fromarray(self.chip.image))
            # self.img = self.img.resize([int(self.zoom * s) for s in self.img.size], Image.ANTIALIAS)
            # self.tkImg = ImageTk.PhotoImage(self.img)
            self.progLabel.config(text="%04d.%d/%04d" % (self.image_idx, self.chip_idx, self.total))

        self.mainPanel.config(width=max(self.tkImg.width(), 400), height=max(self.tkImg.height(), 400))
        self.tkImg.width()
        self.mainPanel.create_image(0, 0, image=self.tkImg, anchor=NW)


        w = self.tkImg.width()
        h = self.tkImg.height()
        #del old ui elements
        for el in self.imgElements:
            self.mainPanel.delete(el)
        self.imgElements = []

        # load labels
        self.clearBBox()
        self.imageName = os.path.split(aereal_image.path)[-1].split('.jpg')[0]
        self.imglbl.config(text='Image: ' + self.imageName)
        if not refresh:
          self.openIr()


        for idx, bbs in enumerate(self.chip.bboxes.bounding_boxes):
            hs = self.api.hsm.get_hs(bbs.hsId)
            # bbs.color = COLORS[bbs.label]
            bbs.color = COLORS[idx]
            if hs.isStatusRemoved():
                if not self.SHOW_UI:
                    continue
                bbs.color = "#FFFFFF"
            elif hs.type == "Duplicate":
                bbs.color = "#ffa500"
            tmpId = self.mainPanel.create_rectangle(self.rint(bbs.x1),
                                                    self.rint(bbs.y1),
                                                    self.rint(bbs.x2),
                                                    self.rint(bbs.y2),
                                                    width=2,
                                                    outline=bbs.color)

            if self.SHOW_UI:
                self.draw_original_center_pt(hs, bbs)
            # paint name
            if not hs.type == "Duplicate" and not hs.isStatusRemoved() and self.SHOW_UI:
                a1 = self.mainPanel.create_text(bbs.x1, bbs.y2, text="status: %s updated: %s" % (hs.status, str(hs.updated)),
                                                anchor="nw",
                                                fill=bbs.color,font="Arial 10 bold")
                r1 = self.mainPanel.create_rectangle(self.mainPanel.bbox(a1), fill="white")
                self.mainPanel.tag_lower(r1, a1) # make text infront of background
                a2 = self.mainPanel.create_text(bbs.x1, bbs.y1-2, text=hs.id, anchor="sw", fill=bbs.color)
                r2 = self.mainPanel.create_rectangle(self.mainPanel.bbox(a2), fill="white")
                self.mainPanel.tag_lower(r2, a2)  # make text infront of background
                self.imgElements.append(r1)
                self.imgElements.append(r2)
                self.imgElements.append(a1)
                self.imgElements.append(a2)

            self.bboxIdList.append(tmpId)
            self.bboxList.append(bbs)
            self.listbox.insert(END, self.bbox_string(bbs))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=bbs.color)
            self.globalhsid = bbs.hsId



        # paint stats
        self.draw_crop_bounds(w,h)
        if aereal_image.fog == "Yes":
            fog_label = self.mainPanel.create_text(w/2, 12, text="FOG",
                                        anchor="ne", fill="red",font="Times 18 bold")
            self.imgElements.append(fog_label)

        if len(self.bboxList) > 0 or not refresh:
            self.listbox.selection_set(0)
        else:
            self.globalhsid = 0



    def saveImage(self):
        header = "id,color_image,thermal_image,hotspot_id,hotspot_type,species_id,species_confidence,fog,thermal_x," \
                 "thermal_y,color_left,color_top,color_right,color_bottom, updated_left, updated_top, updated_right, updated_bottom, " \
                 "updated, status"
        self.api.saveHotspotsToCSV(csv_out, header)



    def append_new_bbox(self, x1, x2, y1, y2):
        hs = self.api.hsm.get_hs(self.globalhsid)
        self.globalhsid = str(round(float(self.globalhsid) + 0.1,1))
        new_hs = HotSpot(self.globalhsid, hs.thermal_loc[0], hs.thermal_loc[1], hs.rgb_bb_l, hs.rgb_bb_t,  hs.rgb_bb_r, hs.rgb_bb_b, hs.type, hs. species, hs.rgb, hs.ir, hs.timestamp,
                         hs.project_name, hs.aircraft, y1, y2, x1, x2, updated=True, status="new")

        self.chip.bboxes.bounding_boxes.append(new_hs.rgb_bb)
        new_box = shift_box(new_hs.rgb_bb, self.chip.crops[2], self.chip.crops[0])
        new_hs.update_bbox(new_box.x1, new_box.y1, new_box.x2, new_box.y2)
        self.api.addHotspot(new_hs)
        self.loadImage(True)

    def update_bbox(self, bbox_idx, x1, y1, x2, y2):
        bbox = self.chip.bboxes.bounding_boxes[bbox_idx]
        hs = self.api.hsm.get_hs(bbox.hsId)

        bbox.x1 = x1
        bbox.x2 = x2
        bbox.y1 = y1
        bbox.y2 = y2
        new_bb = shift_box(bbox, self.chip.crops[2], self.chip.crops[0])
        hs.update_bbox(new_bb.x1, new_bb.y1, new_bb.x2, new_bb.y2)

        self.api.updateHs(hs, True)

        updated_boxes = []
        updated_boxes.append(bbox)
        for idx, bbox in enumerate(self.chip.bboxes.bounding_boxes):
            if bbox.hsId != hs.id:
                updated_boxes.append(bbox)
        self.chip.bboxes = ia.BoundingBoxesOnImage(updated_boxes, shape=self.chip.bboxes.shape)
        self.chips[self.chip_idx] = self.chip

        self.loadImage()

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


    def delBBox_key(self, event=None):
        self.delBBox()

    def clearBBox(self, event=None):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevChip(self, event=None):
        self.saveImage()
        if self.image_idx >= 1:
            if self.chip_idx <= 0:
                self.image_idx -= 1
                self.chip_idx = 0
            else:
                self.chip_idx -= 1
            self.loadImage()
        else:
            self.image_idx = 0
            self.chip_idx = 0
            self.loadImage()

    def nextChip(self, event=None):
        self.saveImage()
        if self.image_idx < self.total:
            if self.chip_idx + 1 >= len(self.chips):
                self.image_idx += 1
                while True:
                    if self.image_idx >= len(self.image_names):
                        break
                    imagePath = self.image_names[self.image_idx - 1]
                    aereal_image = self.api.rgb_images[imagePath]
                    found = False
                    for hotspot in aereal_image.hotspots:
                        if not hotspot.updated and not "duplicate" in hotspot.status:
                            found = True
                    if found:
                        break
                    else:
                        self.image_idx+=1
                self.chip_idx = 0
            else:
                self.chip_idx += 1
            self.loadImage()



    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx <= self.total:
            self.saveImage()
            self.image_idx = idx
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
            self.image_idx = idx
            self.loadImage()

    def setClass(self, event=None):
        self.currentLabelClass = self.className.get()
        print('set label class to :', self.currentLabelClass)

        selection = self.listbox.curselection()

        if len(selection) != 1:
            return
        for sel in selection:
            idx = int(sel)
            hs = self.api.hsm.get_hs(self.bboxList[idx].hsId)
            hs.species = self.currentLabelClass
            hs.classIndex = SpeciesList.index(hs.species)
            self.api.setClass(hs)
            self.api.updateHs(hs, True)
        self.loadImage()

    def zoom_in(self, event=None):
        self.zoom *= 1.2
        self.saveImage()
        self.loadImage()

    def zoom_out(self, event=None):
        self.zoom /= 1.2
        self.saveImage()
        self.loadImage()

    def bbox_string(self, bbox):
        hs = self.api.hsm.get_hs(bbox.hsId)
        other_hs_in_im = [a for a in self.api.rgb_images[hs.rgb.path].hotspots if a.id != hs.id]

        b, t, l, r = hs.getBTLR(True)
        l_orig = l - self.chip.crops[2]
        r_orig = r - self.chip.crops[2]
        t_orig = t - self.chip.crops[0]
        b_orig = b - self.chip.crops[0]
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_y = b_orig + ((t_orig - b_orig) / 2)
        # box2 = self.mainPanel.create_oval(center_x-3 , center_y-3,center_x+3 , center_y+3, width=0, fill='white', outline="#FFF")
        # self.imgElements.append(box2)

        updated_str = "U" if hs != None and hs.updated else "NU"
        return ('%s %s : %s (b:%d, t:%d) (l:%d, r:%d)' % (updated_str, bbox.hsId, bbox.label, b_orig,
                                                          t_orig, l_orig, r_orig))

    def draw_crop_bounds(self,w,h):
        a1 = self.mainPanel.create_text(3, 3, text=("(%d,%d)" % (self.chip.crops[2], self.chip.crops[0])),
                                        anchor="nw", fill="red",font="Times 18 bold")
        a2 = self.mainPanel.create_text(w-3, h-3, text=("(%d,%d)" % (self.chip.crops[3], self.chip.crops[1])),
                                        anchor="se", fill="red",font="Times 18 bold")
        self.imgElements.append(a1)
        self.imgElements.append(a2)

    def draw_original_center_pt(self, hs, bbs):
        b, t, l, r = hs.getBTLR(True)
        l_orig = l - self.chip.crops[2]
        r_orig = r - self.chip.crops[2]
        t_orig = t - self.chip.crops[0]
        b_orig = b - self.chip.crops[0]
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_x = l_orig + ((r_orig - l_orig) / 2)
        center_y = b_orig + ((t_orig - b_orig) / 2)
        box2 = self.mainPanel.create_oval(center_x - 3, center_y - 3, center_x + 3, center_y + 3, width=0,
                                          fill=bbs.color, outline=bbs.color)
        self.imgElements.append(box2)


    def delBBox(self):
        selection = self.listbox.curselection()
        if len(selection) != 1:
            return
        for sel in selection:
            idx = int(sel)
            hs = self.api.hsm.get_hs(self.bboxList[idx].hsId)
            self.api.setStatus(hs, "removed")
            self.api.updateHs(hs, True)
        self.loadImage()

    def toggleUI(self):
        self.SHOW_UI = not self.SHOW_UI
        self.loadImage()

    def set_type(self, type):
        selection = self.listbox.curselection()
        if len(selection) != 1:
            return
        for sel in selection:
            idx = int(sel)
            hs = self.api.hsm.get_hs(self.bboxList[idx].hsId)
            hs.type = type
            self.api.setType(hs, hs.type)  # set hs status to bad res
            self.api.updateHs(hs, True)  # mark hs as updated
        self.loadImage()

    def set_status(self, status):
        selection = self.listbox.curselection()
        if len(selection) != 1:
            return
        for sel in selection:
            idx = int(sel)
            hs = self.api.hsm.get_hs(self.bboxList[idx].hsId)
            if status == "none":
                hs.status = "none"
            else:
                statuses = hs.status.split('-')
                if status in statuses:
                    continue
                if "none" in statuses:
                    statuses.remove('none')
                hs.status = '-'.join(statuses)
                if len(statuses) == 0:
                    hs.status = status
                else:
                    hs.status = hs.status + "-" + status
            self.api.setStatus(hs, hs.status) # set hs status to bad res
            self.api.updateHs(hs, True) # mark hs as updated
        self.loadImage()

    def openIr(self):
        hotspots = self.api.rgb_images[self.image_names[self.image_idx-1]].hotspots
        if len(hotspots) < 1:
            return
        if not hotspots[0].ir.load_image():
            return
        im = hotspots[0].ir.image
        mi = np.percentile(im, 1)
        ma = np.percentile(im, 100)
        normalized = (im - mi) / (ma - mi)

        normalized = normalized * 255
        normalized[normalized < 0] = 0
        normalized = normalized.astype(np.uint8)
        normalized = cv2.resize(normalized, dsize=(320, 256), interpolation=cv2.INTER_CUBIC)
        for child in self.novi.winfo_children():
            child.destroy()
        if not self.novi.winfo_exists():
            self.novi = Toplevel()
        canvas = Canvas(self.novi, width = 320, height = 256)
        canvas.pack(expand = YES, fill = BOTH)
        # gif1 = PhotoImage(file = 'image.gif')
                                    #image not visual
        norm_tk_im = ImageTk.PhotoImage(image=Image.fromarray(normalized))
        canvas.create_image(0, 0, image = norm_tk_im, anchor = NW)
        #assigned the gif1 to the canvas object
        canvas.img = norm_tk_im

class Object(object):
    pass

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.update()
    root.mainloop()
