"""
Numba implementation of envsum operations.
"""
import numba
import numba.cuda
import numpy as np
import torch


from .tensor_wrapper import via_numpy, NumbaCompatibleTensorFunction

# Very basic implementation.
# While simple, it beats a set of pytorch operations simply by using far less memory.

# conventions:
# pidx  : index of pair
# pfidx : index of first atom in pair (receiver)
# psidx : index of second atom in pair (sender)
# fidx  : index of feature
# nidx  : index of sensitivity (nu)

def get_id_and_starts(key):
    key_ids = torch.unique_consecutive(key)
    n_keys = key_ids.shape[0]
    n_items = key.shape[0]
    key_diff = key[:-1] - key[1:]
    key_start = torch.empty(n_keys + 1, dtype=torch.long, device=key.device)
    key_start[1:-1] = torch.nonzero(key_diff, as_tuple=True)[0] + 1
    key_start[0] = 0
    key_start[-1] = n_items
    return key_ids, key_start


def resort_pairs(key, others):
    keysort, argsort = torch.sort(key)
    others = [o[argsort] for o in others]
    key_ids, key_starts = get_id_and_starts(keysort)
    return key_ids, key_starts, keysort, others

# Kernel which sums sensitivities and features to get environment.
# Numpy core signature: (p,n),(a,f),(p),(p),(a,n,f)
class WrappedEnvsum(NumbaCompatibleTensorFunction):
    def __call__(self, *args, **kwargs):
        sense, feat, pfirst, psecond = args
        atom1_ids, atom1_starts, pfirst, (sense, psecond) = resort_pairs(pfirst, [sense, psecond])
        args = sense, feat, pfirst, psecond, atom1_ids, atom1_starts
        return super().__call__(*args, **kwargs)

    def out_shape(self, sense_shape, feat_shape, *other_shapes):
        n_pair, n_nu = sense_shape
        n_atom, n_feature = feat_shape
        return n_atom, n_nu, n_feature

    def launch_bounds(self, sense_shape, fs, pfs, pss, atom1_ids_shape, *other_shapes):
        n_pairs, n_nu = sense_shape
        n_atom, n_feat = fs
        n_atoms_interacting, = atom1_ids_shape
        TPB_MAX = 512
        TPB_X = n_feat
        TPB_Y = TPB_MAX // n_feat
        TPB = (TPB_X, TPB_Y)
        BPG_X = (n_atoms_interacting + TPB_Y - 1) // TPB_Y
        BPG_Y = 1
        BPG = (BPG_X, BPG_Y)
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = 'void({DTYPE}[:,:,],{DTYPE}[:,:],int64[:],int64[:],int64[:],int64[:],{DTYPE}[:,:,:])'.format(
            DTYPE=KERNEL_DTYPE)

        @numba.cuda.jit(sig, )
        def kernel(sens, feat, pfirst, psecond, atom1_ids, atom1_starts, env):
            n_pairs, n_nu = sens.shape
            n_atom, n_feat = feat.shape
            n_atoms_interacting, = atom1_ids.shape

            tx = numba.cuda.threadIdx.x
            sx = numba.cuda.blockDim.x
            by = numba.cuda.blockIdx.y
            fidx = by * sx + tx

            ty = numba.cuda.threadIdx.y
            sy = numba.cuda.blockDim.y
            bx = numba.cuda.blockIdx.x
            i = bx * sy + ty
            if i < n_atoms_interacting:
                pair_start = atom1_starts[i]
                pair_end = atom1_starts[i + 1]
                pfidx = atom1_ids[i]
                for nidx in range(n_nu):
                    out = KERNEL_DTYPE(0.)
                    for pidx in range(pair_start, pair_end):
                        psidx = psecond[pidx]
                        out += sens[pidx, nidx] * feat[psidx, fidx]
                    env[pfidx, nidx, fidx] = out

        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=True)
    def cpu_kernel(sens, feat, pfirst, psecond, atom_ids, atom_starts):

        n_pairs, n_nu = sens.shape
        n_atom, n_feat = feat.shape
        n_atom_with_pairs, = atom_ids.shape

        env_features = np.zeros((n_atom, n_nu, n_feat), dtype=sens.dtype)
        for i in numba.prange(n_atom_with_pairs):
            pfstart = atom_starts[i]
            pfend = atom_starts[i + 1]
            pfidx = atom_ids[i]
            tmp = np.zeros((n_nu, n_feat), dtype=sens.dtype)
            for pidx in range(pfstart, pfend):
                psidx = psecond[pidx]
                for nidx in numba.prange(n_nu):
                    s = sens[pidx, nidx]
                    if abs(s) > 1e-10:
                        for fidx in range(n_feat):
                            tmp[nidx, fidx] += s * feat[psidx, fidx]
            env_features[pfidx] = tmp
        return env_features


# Kernel which sums environment and features to get sensitivity
# Numpy core signature: (a,n,f),(a,f),(p),(p),(p,n),
class WrappedSensesum(NumbaCompatibleTensorFunction):
    def out_shape(self, env_shape, feat_shape, pfirst_shape, psecond_shape):
        n_pair, = pfirst_shape
        n_atom, n_nu, n_feat = env_shape
        return n_pair, n_nu

    def launch_bounds(self, env_shape, feat_shape, pfirst_shape, psecond_shape):
        n_pairs, = pfirst_shape
        TPB = 512
        BPG = (n_pairs + TPB - 1) // TPB
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = 'void({DTYPE}[:,:,:],{DTYPE}[:,:],int64[:], int64[:],{DTYPE}[:,:])'.format(DTYPE=KERNEL_DTYPE)

        @numba.cuda.jit(sig, )
        def kernel(env, feat, pfirst, psecond, sense):
            n_pairs, = pfirst.shape
            n_atom, n_nu, n_feat = env.shape
            pidx, nidx, fidx = numba.cuda.grid(3)

            if pidx < n_pairs:
                pfidx = pfirst[pidx]
                psidx = psecond[pidx]
                for nidx in range(n_nu):
                    tmp = KERNEL_DTYPE(0.)
                    for fidx in range(n_feat):
                        tmp += env[pfidx, nidx, fidx] * feat[psidx, fidx]
                    sense[pidx, nidx] = tmp

        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=True)
    def cpu_kernel(env, feat, pfirst, psecond):
        n_atom, n_nu, n_feat = env.shape
        n_pairs, = pfirst.shape
        sense = np.zeros((n_pairs, n_nu), dtype=env.dtype)
        for pidx in numba.prange(n_pairs):
            pfidx = pfirst[pidx]
            psidx = psecond[pidx]
            for nidx in numba.prange(n_nu):
                tmp = 0
                for fidx in range(n_feat):
                    tmp += env[pfidx, nidx, fidx] * feat[psidx, fidx]
                sense[pidx, nidx] = tmp
        return sense


# Kernel which sums environment and sensitivity to get features
# Numpy core signature: (a,n,f),(p,n),(p),(p),(a,f),
class WrappedFeatsum(NumbaCompatibleTensorFunction):
    def __call__(self, *args, **kwargs):
        env, sense, pfirst, psecond = args
        atom2_ids, atom2_starts, psecond, (sense, pfirst) = resort_pairs(psecond, [sense, pfirst])
        args = env, sense, pfirst, psecond, atom2_ids, atom2_starts
        return super().__call__(*args, **kwargs)

    def out_shape(self, env_shape, *other_shapes):
        n_atom, n_nu, n_feature = env_shape
        return n_atom, n_feature

    def launch_bounds(self, env_shape, sense_shape, pfirst_shape, psecond_shape, atom2_id_shape, atom2_startshape):
        # Note, this kernel has transposed launch bounds:
        # The blocks in the x direction go with threads in the
        # y direction, and likewise for y blocks-> x threads
        # This is done because we want features along the x direction for coherent warps
        # But we want atoms along the block x direction to avoid
        # problems when reaching too many atoms (can only launch 65k blocks in y direction...)
        n_pairs, n_nu = sense_shape
        n_atom, n_nu, n_feat = env_shape
        TPB_max = 512
        if n_feat > 32:
            TPB_x = ((n_feat + 31) // 32) * 32
            TPB_y = TPB_max // TPB_x
            BPG_y = 1
        else:
            # If there aren't enough features to really fill a warp, stride them across the blocks instead.
            TPB_x = 1
            TPB_y = TPB_max
            BPG_y = n_feat
        TPB = (TPB_x, TPB_y)
        BPG_x = (n_atom + TPB_y - 1) // TPB_y
        BPG = (BPG_x, BPG_y)
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = 'void({DTYPE}[:,:,:],{DTYPE}[:,:],int64[:],int64[:],int64[:],int64[:],{DTYPE}[:,:])'.format(
            DTYPE=KERNEL_DTYPE)

        @numba.cuda.jit(sig, )
        def kernel(env, sense, pfirst, psecond, atom2_ids, atom2_starts, feat):
            n_pairs, n_nu = sense.shape
            n_atom, n_feat = feat.shape
            n_atoms_with_pairs, = atom2_ids.shape

            tx = numba.cuda.threadIdx.x
            sx = numba.cuda.blockDim.x
            by = numba.cuda.blockIdx.y
            fidx = by * sx + tx

            ty = numba.cuda.threadIdx.y
            sy = numba.cuda.blockDim.y
            bx = numba.cuda.blockIdx.x
            aidx = bx * sy + ty

            if aidx < n_atoms_with_pairs and fidx < n_feat:
                pair_start = atom2_starts[aidx]
                pair_end = atom2_starts[aidx + 1]
                psidx = atom2_ids[aidx]
                tmp = KERNEL_DTYPE(0.)
                for pidx in range(pair_start, pair_end):
                    pfidx = pfirst[pidx]
                    for nidx in range(n_nu):
                        # env call coalesces across the warp, sense call broadcasts across warp
                        tmp += env[pfidx, nidx, fidx] * sense[pidx, nidx]
                feat[psidx, fidx] = tmp

        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=True)
    def cpu_kernel(env, sens, pfirst, psecond, atom_ids, atom_starts, ):
        n_atom, n_nu, n_feat = env.shape
        n_pairs, = pfirst.shape
        n_atoms_with_pairs, = atom_ids.shape
        feat = np.zeros((n_atom, n_feat), dtype=sens.dtype)

        for i in numba.prange(n_atoms_with_pairs):
            psstart = atom_starts[i]
            psend = atom_starts[i + 1]
            psidx = atom_ids[i]
            tmp = np.zeros(n_feat, dtype=sens.dtype)
            for pidx in range(psstart, psend):
                pfidx = pfirst[pidx]
                for nidx in range(n_nu):
                    for fidx in numba.prange(n_feat):
                        tmp[fidx] += env[pfidx, nidx, fidx] * sens[pidx, nidx]
            feat[psidx] = tmp
        return feat


new_envsum = WrappedEnvsum()
new_sensesum = WrappedSensesum()
new_featsum = WrappedFeatsum()
