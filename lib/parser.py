import struct

def read_vectors(f, parser='fvec'):
    """
    Read vector files asynchronously.
    """
    BLOCK_SIZE = 4096
    WORD_SIZE = 1 if parser=='bvec' else 4
    WORD_TYPE = parser[0]
    f.seek(0)
    raw = f.read(BLOCK_SIZE)
    while len(raw) > 0:
        vec_size = struct.unpack('<i', raw[0:WORD_SIZE])[0]
        while len(raw) >= WORD_SIZE + WORD_SIZE * vec_size:
            s = raw[WORD_SIZE:WORD_SIZE + WORD_SIZE*vec_size]
            yield struct.unpack(f'<{vec_size}{WORD_TYPE}', s)
            raw = raw[WORD_SIZE + WORD_SIZE*vec_size:]
        raw = raw + f.read(BLOCK_SIZE)

def read_vector(f, i, parser='fvec'):
    """
    Read a single vector based on its index in the vec file.
    """
    WORD_SIZE = 1 if parser=='bvec' else 4
    WORD_TYPE = parser[0]
    TUPLE_SIZE = 128
    f.seek(i * TUPLE_SIZE * WORD_SIZE)
    raw = f.read(TUPLE_SIZE * WORD_SIZE)
    return struct.unpack(f'<{TUPLE_SIZE}{WORD_TYPE}', raw)