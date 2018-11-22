from ConfigParser import SafeConfigParser

from arcticapi.crop import CropCfg

configKeys = ['csv', 'imdir', 'imout', 'bbox_size', 'min_shift', 'max_shift',
                      'crop_size', 'merge_seal_classes', 'make_bear', 'make_anomaly', 'debug']

cfgFileName = 'config.ini'
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

    with open(cfgFileName, 'w') as configfile:
        config.write(configfile)
    return True


def load_config(name):
    config = SafeConfigParser()
    config.read(cfgFileName)
    if not config.has_section(name):
        print("")
        return None

    print("["+name+"]")
    for candidate in configKeys:
        if not config.has_option(name, candidate):
            print("Invalid config, missing option: " + candidate)
            return None
        print('%-12s: %s' % (candidate, config.get(name, candidate)))
    print("")

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
    return CropCfg(csv, imdir, imout, bbox_size, min_shift, max_shift, crop_size, "training_list.txt", merge_seal_classes,
                   make_bear, make_anomaly, debug)

