from subprocess import PIPE

from psutil import Popen

p = Popen(["python", "-u", "sp.py"], stdin=PIPE, stdout=PIPE, bufsize=1)
print p.stdout.readline(), # read the first line
for i in range(10): # repeat several times to show that it works
    print >>p.stdin, i # write input
    p.stdin.flush() # not necessary in this case
    print p.stdout.readline(), # read output

print p.communicate("n\n")[0], # signal the child to exit,
                               # read the rest of the output,
                               # wait for the child to exit