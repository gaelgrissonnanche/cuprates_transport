import argparse
import sys

def easy_args(default_args, help_dict=None, description=None, usage=None, argv=sys.argv, out_dict=False,):
    parser = argparse.ArgumentParser(description=description, usage=usage)
    for name, default in default_args.items():
        helpstr = ""
        try:
            helpstr = help_dict[name]
        except (KeyError, NameError, TypeError):
            pass

        if type(default) is list:
            parser.add_argument("--"+name, nargs="+", type=type(default[0]), default=default, help=helpstr)
        elif type(default) is bool and default is True:
            parser.add_argument("--no_"+name, dest=name, action="store_false", help=f"disables {name}")
        elif type(default) is bool and default is False:
            parser.add_argument("--"+name, action="store_true", default=default, help=helpstr)
        elif default is None:
            parser.add_argument("--"+name, default=default, help=helpstr)
        else:
            parser.add_argument("--"+name, type=type(default), default=default, help=helpstr)

    if out_dict:
        return vars(parser.parse_known_args(argv)[0])
    else:
        return parser.parse_known_args(argv)[0]

def args_name(params_dict, default_dict, ignore_list=[]):
    filename = params_dict['bandname']
    for name, value in params_dict.items():
        default_value = default_dict[name]
        if name in ignore_list:
            pass
        elif value != default_value:
            filename += f"_{name}{value}"
    return filename