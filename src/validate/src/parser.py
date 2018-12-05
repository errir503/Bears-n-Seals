import csv


class Parser():
    def __init__(self, csv_file):
        rows = []
        f = open(csv_file, 'r')
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
        f.close()

        row_objects = []
        d = {'SEAL': 0, 'NOTSEAL': 0, 'MAYBESEAL': 0, 'UNCHECKED': 0}
        for row in rows:
            status = row[11]
            d[status] += 1
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
                obj.status = 'UNCHECKED'
            else:
                status = row[11]
                obj.status = status
            row_objects.append(obj)
        del rows
        print(d)
        self.rows = row_objects

    def get_objects(self):
        return self.rows