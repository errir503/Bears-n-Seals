from ConfigParser import SafeConfigParser

from arcticapi.crop import CropCfg


def make_config(args):
    config = SafeConfigParser()
    config.read('config.ini')
    if config.has_section(args.name):
        config.remove_section(args.name)

    name = args.name
    config.add_section(name)
    config.set(name, 'csv', args.csv)
    config.set(name, 'imdir', args.imdir)
    config.set(name, 'imout', args.imout)
    config.set(name, 'bbox_size', str(args.bb))
    config.set(name, 'min_shift', str(args.min))
    config.set(name, 'max_shift', str(args.max))
    config.set(name, 'crop_size', str(args.cs))
    config.set(name, 'merge_seal_classes', str(args.c))
    config.set(name, 'make_bear', str(args.b))
    config.set(name, 'make_anomaly', str(args.a))
    config.set(name, 'debug', str(args.d))

    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    return True


def get_config(name):
    config = SafeConfigParser()
    config.read('config.ini')
    if not config.has_section(name):
        return None

    for candidate in ['csv', 'imdir', 'imout', 'bbox_size', 'min_shift', 'max_shift',
                      'crop_size', 'merge_seal_classes', 'make_bear', 'make_anomaly', 'debug']:
        if not config.has_option(name, candidate):
            print("Invalid config, missing option: " + candidate)
            return None
        print('%-12s: %s' % (candidate, config.get(name, candidate)))
    print("")

    csv = config.get(name, 'csv')
    imdir = config.get(name, 'imdir')
    imout = config.get(name, 'imout')
    bbox_size = config.getint(name, 'bbox_size')
    min_shift = config.getint(name, 'min_shift')
    max_shift = config.getint(name, 'max_shift')
    crop_size = config.getint(name, 'crop_size')
    merge_seal_classes = config.getboolean(name, 'merge_seal_classes')
    make_bear = config.getboolean(name, 'make_bear')
    make_anomaly = config.getboolean(name, 'make_anomaly')
    debug = config.getboolean(name, 'debug')
    return CropCfg(csv, imdir, imout, bbox_size, min_shift, max_shift, crop_size, "training_list.txt", merge_seal_classes,
                   make_bear, make_anomaly, debug)

