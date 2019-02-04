from ConfigParser import SafeConfigParser

from visuals import bcolors

configKeys = ['csv', 'imdir', 'imout', 'bbox_size', 'min_shift', 'max_shift',
                      'crop_size', 'merge_seal_classes', 'make_bear', 'make_anomaly', 'debug', 'image_type', 'output_list', 'merge_all_classes']

cfgFileName = 'config.ini'

class CropCfg(object):
    def __init__(self, csv, im_dir, out_dir, bbox_size, minShift, maxShift, crop_size, label, combine_seal, make_bear, make_anomaly,
                 debug, imtype, name, combine_all):
        self.csv = csv
        self.im_dir = im_dir
        self.out_dir = out_dir
        self.bbox_size = bbox_size
        self.minShift = minShift
        self.maxShift = maxShift
        self.crop_size = crop_size
        self.label = label
        self.combine_seal = combine_seal
        self.make_bear = make_bear
        self.make_anomaly = make_anomaly
        self.combine_all = combine_all
        self.debug = debug
        self.imtype = imtype
        self.name = name

def make_model_config(cfg, classes):
    config = SafeConfigParser()
    config.read(cfgFileName)
    name = cfg.name + "_model"

    if config.has_section(name):
        config.remove_section(name)

    config.add_section(name)

    config.set(name, "classes", str(classes))
    config.set(name, "width", str(cfg.crop_size))
    config.set(name, "height", str(cfg.crop_size))

    with open(cfgFileName, 'w') as configfile:
        config.write(configfile)
    return True



def make_config(args):
    config = SafeConfigParser()
    config.read(cfgFileName)
    if config.has_section(args.name):
        config.remove_section(args.name)

    name = args.name
    config.add_section(name)
    config.set(name, configKeys[0], args.csv)
    config.set(name, configKeys[1], args.imdir)
    config.set(name, configKeys[2], args.imout)
    config.set(name, configKeys[3], str(args.bb))
    config.set(name, configKeys[4], str(args.min))
    config.set(name, configKeys[5], str(args.max))
    config.set(name, configKeys[6], str(args.cs))
    config.set(name, configKeys[7], str(args.c))
    config.set(name, configKeys[8], str(args.b))
    config.set(name, configKeys[9], str(args.a))
    config.set(name, configKeys[10], str(args.d))
    config.set(name, configKeys[11], args.imtype)
    config.set(name, configKeys[12], args.outlist)
    config.set(name, configKeys[13], args.all)

    with open(cfgFileName, 'w') as configfile:
        config.write(configfile)
    return True


def load_config(name):
    config = SafeConfigParser()
    config.read(cfgFileName)
    if not config.has_section(name):
        print("Config " + name + " was not found.")
        return None

    print("["+name+"]")
    for candidate in configKeys:
        if not config.has_option(name, candidate):
            print("Invalid config, missing option: " + candidate)
            return None

        val = config.get(name, candidate)

        if candidate == "debug":
            print('%-12s: %s%s%s' % (candidate, bcolors.OKBLUE if val == 'False' else bcolors.RED, val, bcolors.ENDC))
        else:
            print('%-12s: %s' % (candidate, val))
    print
    csv = config.get(name, configKeys[0])
    imdir = config.get(name, configKeys[1])
    imout = config.get(name, configKeys[2])
    bbox_size = config.getint(name, configKeys[3])
    min_shift = config.getint(name, configKeys[4])
    max_shift = config.getint(name, configKeys[5])
    crop_size = config.getint(name, configKeys[6])
    merge_seal_classes = config.getboolean(name, configKeys[7])
    make_bear = config.getboolean(name, configKeys[8])
    make_anomaly = config.getboolean(name, configKeys[9])
    debug = config.getboolean(name, configKeys[10])
    imtype = config.get(name, configKeys[11])
    outlist = config.get(name, configKeys[12])
    combine_all = config.getboolean(name, configKeys[13])
    return CropCfg(csv, imdir, imout, bbox_size, min_shift, max_shift, crop_size, outlist, merge_seal_classes,
                   make_bear, make_anomaly, debug, imtype, name, combine_all)

