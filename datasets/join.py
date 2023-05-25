import sys, os
from utils import join_files

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '-help':
        print('Use: join.py [from-dir-name] [to-file-name]')
    else:
        if len(sys.argv) != 3:
            interactive = 1
            fromdir = input('Directory Containing Part files: ')
            tofile  = input('Output FileName: ')
        else:
            interactive = 0
            fromdir, tofile = sys.argv[1:]
        absfrom, absto = map(os.path.abspath, [fromdir, tofile])
        print('Joining %s to make %s' % (absfrom, absto))

        try:
            join_files(fromdir, tofile)
        except:
            print('Error joining files')
        else:
           print('Join complete: see', absto)
        if interactive: input('Press Enter key') # pause if clicked
