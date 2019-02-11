import imgaug as ia


class BoundingBox(ia.BoundingBox):
    def __init__(self, l, t, r, b, label, hsid):
        ia.BoundingBox.__init__(self, x1=l, y1=t, x2=r, y2=b, label=label)
        self.hsId = hsid
