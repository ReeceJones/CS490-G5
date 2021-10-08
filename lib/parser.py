import struct

def read_vectors(f, parser='fvec'):
    """
    Read vector files asynchronously.
    """
    BLOCK_SIZE = 4096
    WORD_SIZE = 1 if parser=='bvec' else 4
    WORD_TYPE = parser[0]
    raw = f.read(BLOCK_SIZE)
    while len(raw) > 0:
        vec_size = struct.unpack('<i', raw[0:WORD_SIZE])[0]
        while len(raw) >= WORD_SIZE + WORD_SIZE * vec_size:
            s = raw[WORD_SIZE:WORD_SIZE + WORD_SIZE*vec_size]
            yield struct.unpack(f'<{vec_size}{WORD_TYPE}', s)
            raw = raw[WORD_SIZE + WORD_SIZE*vec_size:]
        raw = raw + f.read(BLOCK_SIZE)

