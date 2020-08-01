import os
import glob


def flush_directory(path, exclude=None):
    if exclude:
        files = glob.glob(path + '*[!.{}]'.format(exclude))
    else:
        files = glob.glob(path + '*')

    if len(files) == 0:
        print("Directory is already flushed.")
    else:
        for f in files:
            print('{} successfully removed'.format(f))
            os.remove(f)
