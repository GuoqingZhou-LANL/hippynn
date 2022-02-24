import os
#"""
alist = []
with open("info.txt") as f:
    for l in f:
        alist.append(int(l.strip().split()[0]))
#"""
#alist = [0]
alist = alist + list(range(alist[-1]+1,10000))

for c in alist:
    if os.path.exists("./files/%d/log" % c):
        #print(c)
        if len(os.popen("grep 'SCF NOT CONVERGED AFTER 125 CYCLES' ./files/%d/log" % c).read()) != 0:
            print("%5d SCF" % c)
            #os.system("rm ./files/%d/*.tmp ./files/%d/mol.gbw" % (c,c))
            continue

        #if len(os.popen("grep 'SCF NOT CONVERGED AFTER 500 CYCLES' ./files/%d/log" % c).read()) != 0:
        #    print("%5d SCF 500" % c)
        #    os.system("rm ./files/%d/*.tmp ./files/%d/mol.gbw" % (c,c))
        #    continue

        if len(os.popen("grep 'THE CP-SCF CALCULATION IS UNCONVERGED' ./files/%d/log" % c).read()) != 0:
            print("%5d CP-SCF" % c)
            #os.system("rm ./files/%d/*.tmp ./files/%d/mol.gbw" % (c,c))
            continue

        #if len(os.popen("grep 'SERIOUS PROBLEM IN SOSCF' ./files/%d/log" % c).read()) != 0:
        #    print("%5d SOSCF" % c)
        #    os.system("rm ./files/%d/*.tmp ./files/%d/mol.gbw" % (c,c))
        #    continue

        ss = os.popen("tail -n 2 ./files/%d/log | head -n 1" % c).read().strip()
        if "aborting the run" in ss or "[COORDS]" in ss:
            print("%5d error" % c)
            os.system("rm -rf ./files/%d" % c)
        elif "TOTAL RUN TIME: 0 days" not in os.popen("tail -n 1 ./files//%d/log " % c).read().strip():
            print("%5d not done" % c)
            #os.system("rm ./files/%d/*.tmp ./files/%d/mol.gbw" % (c,c))
            #os.system("rm  ./files/%d/log" % c)
 
