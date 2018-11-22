from string import Template

model_name = "testmodel"
model_directory = "testdir"


def makeData():
    d = {}
    d['classes'] = 1
    d['train'] = model_name + "_train.txt"
    d['valid'] = model_name + "_valid.txt"
    d['labels'] = model_name + ".labels"
    d['names'] = model_name + ".names"
    d['backup'] = model_name + "_backup/"
    d['results'] = model_name + "_results/"

    # open the file
    filein = open('src/models/data_template.data')
    # read it
    src = Template(filein.read())
    # do the substitution
    res = src.substitute(d)


    text_file = open(model_name + ".data", "w")
    text_file.write(res)
    text_file.close()

def makeCfg():
    d = {}
    d['classes'] = 1
    d['filters'] = (d['classes'] + 5) * 3
    d['width'] = 800
    d['height'] = 800

    # open the file
    filein = open('src/models/yolov3.cfg')
    # read it
    src = Template(filein.read())
    # do the substitution
    res = src.substitute(d)

    print(res)

makeData()

