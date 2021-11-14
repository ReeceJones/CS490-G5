import os
import struct
from typing import List, Set, Tuple

class FSGraph:
    """
    Disk-based graph structure. Index is stored in memory.

    File structure:
        | Header | Graph Index | Graph Data |

    Header:
        | R | Num_points | Tuple_size | Data_type |
    R = 64-bit int
    Num_points = 64-bit int
    Tuple_size = 64-bit int
    Data_type = 4-byte string. 'fvec', 'bvec', 'ivec'.

    Graph Index:
        | Point_1 | ... | Point_n |

    Point:
        | Neighbors |
    Nighbors = R 64-bit integers. All points must be indexed > 0.

    Graph Data:
        | Tuple_1 | ... | Tuple_n |

    """
    def __init__(self, path: str):
        """
        Open a possibly existing FSGraph file
        """
        self.path = path
        self.initialized = False
        self.index = list()
        if os.path.exists(path):
            # attempt to load FSGraph file
            with open(path, 'rb') as self.fd:
                # read header
                nbytes = 8 * 3 + 4
                self.fd.seek(0)
                b = self.fd.read(nbytes)
                header = struct.unpack('<3Q4s', b)
                self.R = header[0]
                self.num_points = header[1]
                self.tuple_size = header[2]
                self.data_type = header[3].decode('ascii')[:4]
                # load index
                self.index = [None for i in range(self.num_points)]
                for i in range(self.num_points):
                    b = self.fd.read(self.R * 8)
                    neighbors = {x for x in struct.unpack(f'<{self.R}Q', b) if x > 0}
                    self.index[i]=neighbors
                self.initialized = True

            if self.initialized:
                self.fd = open(path, 'rb')

    def __del__(self):
        if self.initialized:
            self.fd.close()
        del self.index

    def new(self, R: int, tuple_size: int, data_type: str, pre_allocate_size: int = 0) -> bool:
        """
        Create a new FSGraph File
        """
        if self.initialized:
            self.fd.close()
        self.fd = open(self.path, 'wb')
        self.R = R
        self.tuple_size = tuple_size
        self.data_type = data_type[:4]
        self.num_points = pre_allocate_size

        # write header to file
        self.fd.write(struct.pack('<3Q4s', self.R, self.num_points, self.tuple_size, bytes(self.data_type[:4], 'ascii')))
        
        # pre-allocate space as needed
        if pre_allocate_size > 0:
            self.fd.write(b'\x00' * pre_allocate_size * tuple_size * (1 if data_type[0] == 'b' else 4))

        return True


    def get_data(self, idx: int) -> Tuple[int]:
        HEADER_SIZE = 8 * 3 + 4
        INDEX_SIZE = self.R * 8 * self.num_points
        START_INDEX = HEADER_SIZE + INDEX_SIZE
        WORD_SIZE = 1 if self.data_type=='bvec' else 4
        WORD_TYPE = self.data_type[0]

        if idx >= len(self.index):
            return None

        self.fd.seek(START_INDEX + (idx * self.tuple_size * WORD_SIZE))
        data = struct.unpack(f'<{self.tuple_size}{WORD_TYPE}', self.fd.read(self.tuple_size * WORD_SIZE))

        return data

    def set_data(self, idx: int, data: Tuple[int]) -> int:
        HEADER_SIZE = 8 * 3 + 4
        INDEX_SIZE = self.R * 8 * self.num_points
        START_INDEX = HEADER_SIZE + INDEX_SIZE
        WORD_SIZE = 1 if self.data_type=='bvec' else 4
        WORD_TYPE = self.data_type[0]

        if idx >= len(self.index):
            self.index.append(set())

        self.fd.seek(START_INDEX + (idx * self.tuple_size * WORD_SIZE))
        self.fd.write(struct.pack(f'<{self.tuple_size}{WORD_TYPE}', *data))

        return len(self.index) - 1

    def get_neighbors(self, idx: int) -> Set[int]:
        if idx >= len(self.index):
            return None
        return self.index[idx]

    def __getitem__(self, idx: int) -> Set[int]:
        return self.get_neighbors(idx)

    def set_neighbors(self, idx: int, neighbors: Set[int]) -> bool:
        HEADER_SIZE = 8 * 3 + 4
        START_INDEX = HEADER_SIZE + self.R * 8 * idx
        WORD_SIZE = 1 if self.data_type=='bvec' else 4
        WORD_TYPE = self.data_type[0]

        if idx > len(self.index):
            return False

        # update memory-index
        self.index[idx] = neighbors

        # update disk-index
        self.fd.seek(START_INDEX)
        ww = tuple(list(neighbors) + [-1] * (10-len(neighbors)))
        self.fd.write(struct.pack(f'<{self.R}Q', *ww))

        return True

    def __setitem__(self, idx: int, neighbors: Set[int]) -> bool:
        return self.set_neighbors(idx, neighbors)
