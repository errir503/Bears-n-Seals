# # import subprocess
# #
# # process = subprocess.Popen(cmd, shell=True,
# #                            stdout=subprocess.PIPE,
# #                            stderr=subprocess.PIPE)
# #
# # # wait for the process to terminate
# # out, err = process.communicate()
# # errcode = process.returncode
# from subprocess import Popen, PIPE, STDOUT
#
# p = Popen(['grep', 'f'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
# grep_stdout = p.communicate(input=b'one\ntwo\nthree\nfour\nfive\nsix\n')[0]
# print(grep_stdout.decode())
# # import sys
# # while True:
# #     line = sys.stdin.readline()
# #     if line == "terminate":
# #         break
# #     print line
# #     # do something with line
# #
# #
# # print("process completed")