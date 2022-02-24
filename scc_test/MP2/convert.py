import numpy as np
import sys
fn =  sys.argv[1]

traceless = False
try:
    sys.argv[2] == "traceless"
    traceless = True
except:
    pass

x = np.load(fn)

diag = np.concatenate([x[:,0:1,0], x[:,1:2,1], x[:,2:3,2]], axis=1)
offdiag = np.concatenate([x[:,0:1,1], x[:,1:2,2], x[:,2:3,0]], axis=1)

if traceless:
    diag -= diag.mean(axis=1).reshape(-1,1)

np.save(fn[:-4]+"_offdiag.npy", offdiag)
if traceless:
    np.save(fn[:-4]+"_diag_traceless.npy", diag)
else:
    np.save(fn[:-4]+"_diag.npy", diag)

print(diag.shape)
print(offdiag.shape)
if traceless:
    print(np.abs(diag.sum(axis=1)).max())
