import csv
import time
from Tkinter import *
from PIL import Image, ImageTk, ImageDraw
import os

COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
STATUS_COLOR = ['green', 'red', 'yellow', 'black']
STATUS = ['SEAL', 'NOTSEAL', 'MAYBESEAL', 'UNCHECKED']

class LabelTool():
    #---------------------------------------------------------------------------
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = True, height = True)

        # initialize global state
        self.csv_file = "new_file.csv"
        self.detection_list= self.get_detections()
        self.bboxList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.tkimg = None
        self.img = None
        self.bbox = None

        # ----------------- GUI stuff ---------------------
        # main panel for labeling
        self.mainPanel = Canvas(self.frame)
        self.parent.bind("a", self.prevImage) # press 'a' to go backforward
        self.parent.bind("d", self.nextImage) # press 'd' to go forward
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        # self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        # self.lb1.grid(row = 1, column = 2,  sticky = W+N)
        # self.listbox = Listbox(self.frame, width = 22, height = 12)
        # self.listbox.grid(row = 2, column = 2, sticky = N)

        self.btnMarkSeal = Button(self.frame, text ='Seal!', command = self.mark_seal)
        self.btnMarkSeal.grid(row = 1, column = 2, sticky =W + E + N)
        self.btnMarkMaybe = Button(self.frame, text ='Maybe Seal?', command = self.mark_maybe)
        self.btnMarkMaybe.grid(row = 2, column = 2, sticky =W + E + N)
        self.btnMarkNot = Button(self.frame, text ='Not Seal', command = self.mark_no)
        self.btnMarkNot.grid(row = 3, column = 2, sticky =W + E + N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 6, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next Unchecked >>', width = 10, command = self.nextUncheckedImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)
        self.loadDir()

    #---------------------------------------------------------------------------
    def loadDir(self):
        self.parent.focus()

        # get image list
        if len(self.detection_list) == 0:
            print 'No rows were found in the csv file!'
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.detection_list)

         # set up output dir
        self.outDir = '.'
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)

        self.loadImage()
        print '%d images loaded' %(self.total)

    #---------------------------------------------------------------------------
    def loadImage(self):
        # load image
        start = time.time()
        row = self.detection_list[self.cur - 1]
        img = row.img
        if img is None:
            img = self.load_crop_img(row)

        self.img = img

        self.tkimg = row.tkimg
        if self.tkimg is None:
            self.tkimg = ImageTk.PhotoImage(self.img)

        # self.tkimg = tkimg

        self.disp.config(text= "Prediction: " +str(row.pred))
        p0 = (row.local_x -row.bbox_width/2, row.local_y-row.bbox_height/2)
        p1 = (row.local_x + row.bbox_width/2, row.local_y+row.bbox_height/2)
        s_idx = STATUS.index(row.status)

        width, height = self.img.size
        self.mainPanel.config(width = max(width, 400), height = max(height, 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        self.bbox = self.mainPanel.create_rectangle(p0[0], p0[1], \
                                        p1[0], p1[1], \
                                        width=2, \
                                        outline=STATUS_COLOR[s_idx])
        elapsed = time.time() - start
        print elapsed
        self.chache_next()


    #---------------------------------------------------------------------------
    def chache_next(self):
        if self.cur < len(self.detection_list):
            print("Caching")
            img = self.load_crop_img(self.detection_list[self.cur])
            self.detection_list[self.cur].img = img
            self.detection_list[self.cur].tkimg = ImageTk.PhotoImage(img)


        if self.cur - 2 > 0:
            print("Can't free prev")
            self.detection_list[self.cur-2].img = None #FREE
            self.detection_list[self.cur-2].tkimg = None #FREE

    #---------------------------------------------------------------------------
    def load_crop_img(self, row):
        img = Image.open(row.file)
        img = img.crop((row.crop_left, row.crop_top, row.crop_right, row.crop_bot))
        return img
    #---------------------------------------------------------------------------
    def mark_seal(self):
        self.detection_list[self.cur - 1].status = STATUS[0]
        self.nextImage()

    #---------------------------------------------------------------------------
    def mark_no(self):
        self.detection_list[self.cur - 1].status = STATUS[1]
        self.nextImage()

    #---------------------------------------------------------------------------
    def mark_maybe(self):
        self.detection_list[self.cur - 1].status = STATUS[2]
        self.nextImage()


    #---------------------------------------------------------------------------
    def updateDetection(self):
        self.mainPanel.delete(self.bbox)
        header="fnum,file_name,prediction,local_x,local_y,bbox_width,bbox_height,crop_top,crop_bot,crop_left,crop_right,status\n"
        with open('new_file.csv', 'w') as temp_file:
            temp_file.write(header)
            for row in self.detection_list:
                newrowtxt = ",".join([str(row.num), row.file,str(row.pred), str(row.local_x),
                                      str(row.local_y),str(row.bbox_width),str(row.bbox_height),str(row.crop_top),str(row.crop_bot),str(row.crop_left),
                                      str(row.crop_right),row.status]) + "\n"
                temp_file.write(newrowtxt)

    #---------------------------------------------------------------------------
    def prevImage(self, event = None):
        self.updateDetection()
        if self.cur > 1:
            self.img = None
            self.tkimg = None
            self.cur -= 1
            self.loadImage()

    #---------------------------------------------------------------------------
    def nextImage(self, event = None):
        self.updateDetection()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()
    #---------------------------------------------------------------------------
    def nextUncheckedImage(self, event = None):
        self.updateDetection()
        i = 0
        for det in self.detection_list:
            if det.status == "UNCHECKED":
                break
            i+=1

        if i < self.total:
            self.img = None
            self.tkimg = None
            self.cur = i+1
            self.loadImage()

    #---------------------------------------------------------------------------
    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.updateDetection()
            self.cur = idx
            self.loadImage()

    #---------------------------------------------------------------------------
    def get_detections(self):
        rows = []
        f = open(self.csv_file, 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()
        del rows[0]  # remove col headers
        row_objects = []

        for row in rows:
            obj = lambda: None
            obj.num = int(row[0])
            obj.file = row[1]
            obj.pred = float(row[2])
            obj.local_x = float(row[3])
            obj.local_y = float(row[4])
            obj.bbox_width = float(row[5])
            obj.bbox_height = float(row[6])
            obj.crop_top = int(row[7])
            obj.crop_bot = int(row[8])
            obj.crop_left = int(row[9])
            obj.crop_right = int(row[10])
            if len(row) != 12:
                obj.status = STATUS[3]
            else:
                status = row[11]
                obj.status = status
            row_objects.append(obj)
            obj.img = None
            obj.tkimg = None
        del rows
        return row_objects
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()