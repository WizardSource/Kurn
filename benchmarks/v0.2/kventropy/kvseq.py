"""Parse a llama_state_seq_save_file blob (single stream, no cell ext) into per-layer K / V row arrays."""
import numpy as np

TYPES = {1: "f16", 2: "q4_0", 8: "q8_0", 43: "k4c"}


class Seq:
    def __init__(self, path):
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        off = 0

        def u32():
            nonlocal off
            v = int(self.mm[off:off + 4].view(np.uint32)[0]); off += 4; return v

        def u64():
            nonlocal off
            v = int(self.mm[off:off + 8].view(np.uint64)[0]); off += 8; return v

        magic, ver, ntok = u32(), u32(), u32()
        self.tokens = np.array(self.mm[off:off + 4 * ntok].view(np.int32)); off += 4 * ntok
        nstream = u32(); assert nstream == 1
        n = u32(); self.n = n
        pos = []
        for _ in range(n):
            p = int(self.mm[off:off + 4].view(np.int32)[0]); nsid = int(self.mm[off + 4:off + 8].view(np.uint32)[0])
            off += 8 + 4 * nsid; pos.append(p)
        self.pos = np.array(pos)
        vtrans, nlayer = u32(), u32(); assert vtrans == 0
        self.K, self.V, self.ktype, self.vtype = [], [], None, None
        for lst, attr in ((self.K, "ktype"), (self.V, "vtype")):
            for _ in range(nlayer):
                t = u32(); row = u64()
                setattr(self, attr, TYPES[t])
                lst.append(self.mm[off:off + n * row].reshape(n, row)); off += n * row
        self.nlayer = nlayer
        assert off == len(self.mm), (off, len(self.mm))
