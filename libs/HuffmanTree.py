from queue import PriorityQueue

## Normalized Huffman Tree(BestTree)
class HuffmanTree:
    class __Node:
        def __init__(self, value, freq, left_child, right_child):
            self.value = value
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child

        @classmethod
        def init_leaf(self, value, freq):
            return self(value, freq, None, None)

        @classmethod
        def init_node(self, left_child, right_child):
            freq = left_child.freq + right_child.freq
            return self(None, freq, left_child, right_child)

        def is_leaf(self):
            return self.value is not None

        def __eq__(self, other):
            stup = self.value, self.freq, self.left_child, self.right_child
            otup = other.value, other.freq, other.left_child, other.right_child
            return stup == otup

        def __nq__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return self.freq < other.freq

        def __le__(self, other):
            return self.freq < other.freq or self.freq == other.freq

        def __gt__(self, other):
            return not (self <= other)

        def __ge__(self, other):
            return not (self < other)

    def __init__(self, freq_dict):
        q = PriorityQueue()
        self.huffmanTree = dict() # key, (freq, codeword_len)
        # calculate frequencies and insert them into a priority queue
        for val, freq in freq_dict.items():
            self.huffmanTree[val] = [freq]
            q.put(self.__Node.init_leaf(val, freq))

        while q.qsize() >= 2:
            u = q.get()
            v = q.get()

            q.put(self.__Node.init_node(u, v))

        self.__root = q.get()

        # dictionaries to store huffman table
        self.__value_to_bitstring = dict()

        self.calc_node_codeLen(self.__root)
        # print("self.huffmanTree", self.huffmanTree)

    def calc_node_codeLen(self, n, codelen=0):
        if n.value is not None:
            self.huffmanTree[n.value].append(codelen)
        else:
            self.calc_node_codeLen(n.left_child, codelen + 1)
            self.calc_node_codeLen(n.right_child, codelen + 1)

    def sorted(self):
        HuffmanTreeNodes = list(self.huffmanTree.items())
        HuffmanTreeNodes.sort(key=lambda x: x[1][1]) ## sorted freq of arr
        HuffmanTreeNodes.sort(key=lambda x: x[1][0], reverse=True) ## sorted len of code
        return HuffmanTreeNodes

    ## Create the Canonical Huffman Code by self.huffmanTree
    def create_huffman_table(self):
        HuffmanTreeNodes = self.sorted()

        HuffmanTable = dict()
        code_len = HuffmanTreeNodes[0][1][1]
        code_val = 0b0
        
        for i in range(len(HuffmanTreeNodes)):
            key, (codeFreq, codeLen) = HuffmanTreeNodes[i]
            if codeLen != code_len:
                code_val <<= (codeLen-code_len)
                code_len = codeLen
            elif (i == len(HuffmanTreeNodes)-1) and code_len < 16:
                code_val = code_val << 1 if code_len < 16 else code_val+1
            HuffmanTable[key] = "{:0{}b}".format(code_val, codeLen)
            code_val += 1
        return HuffmanTable

    def mostLongCodeLen(self):
        return max(self.huffmanTree.values(), key=lambda x: x[1])[1]

    def compressed_size(self):
        compressed_size = 0
        for val, (freq, codeLen) in self.huffmanTree.items():
            compressed_size += freq * codeLen
        return compressed_size

    @classmethod
    def init_arr(self, arr):
        return self(self.__calc_freq(arr))

    @classmethod
    def __calc_freq(self, arr):
        freq_dict = dict()
        for elem in arr:
            if elem in freq_dict:
                freq_dict[elem] += 1
            else:
                freq_dict[elem] = 1
        return freq_dict



## LimitedLenHuffmanTree
class LimitedLenHuffmanTree:
    class __Node:
        def __init__(self, value, freq, left_child, right_child):
            self.value = value
            self.freq = freq
            self.left_child = left_child
            self.right_child = right_child

        @classmethod
        def init_leaf(self, value, freq):
            return self(value, freq, None, None)

        @classmethod
        def init_node(self, left_child, right_child):
            freq = left_child.freq + right_child.freq
            return self(None, freq, left_child, right_child)

        def is_leaf(self):
            return self.value is not None

        def __eq__(self, other):
            stup = self.value, self.freq, self.left_child, self.right_child
            otup = other.value, other.freq, other.left_child, other.right_child
            return stup == otup

        def __nq__(self, other):
            return not (self == other)

        def __lt__(self, other):
            return self.freq < other.freq

        def __le__(self, other):
            return self.freq < other.freq or self.freq == other.freq

        def __gt__(self, other):
            return not (self <= other)

        def __ge__(self, other):
            return not (self < other)

    def __init__(self, freq_dict):
        q = PriorityQueue()
        self.total_freq = dict()
        # calculate frequencies and insert them into a priority queue
        for val, freq in freq_dict.items():
            self.total_freq[val] = [freq]
            q.put(self.__Node.init_leaf(val, freq))

        self.m = q.qsize()
        run_index = 0
        while(q.qsize() < (2*self.m-2)):
            # print("# i={}".format(run_index))
            run_index += 1
            newQ = PriorityQueue()
            for _ in range((q.qsize()//2)):
                u = q.get()
                v = q.get()
                newN = self.__Node.init_node(u, v)
                newQ.put(newN)
            for val, freq in freq_dict.items():
                newQ.put(self.__Node.init_leaf(val, freq))
            q = newQ

        self.LimitLengthQuere = q

        self.getSymbolCodeLength()
        for val, freq in self.codeLength_dict.items():
            self.total_freq[val].append(freq)

    @classmethod
    def init_arr(self, arr):
        return self(self.__calc_freq(arr))

    def getSymbolCodeLength(self):
        self.codeLength_dict = dict()
        while self.LimitLengthQuere.qsize() > 0:
            n = self.LimitLengthQuere.get()
            self.__calc_QuereFreq(n)

        return self.codeLength_dict

    def __calc_QuereFreq(self, n):
        if n.value is not None:
            sym = str(n.value)
            if sym in self.codeLength_dict:
                self.codeLength_dict[sym] += 1
            else:
                self.codeLength_dict[sym] = 1
        else:
            self.__calc_QuereFreq(n.left_child)
            self.__calc_QuereFreq(n.right_child)

    ## Create the Canonical Huffman Code by self.huffmanTree
    def create_huffman_table(self):
        huffman_total_table = list(self.total_freq.items())
        huffman_total_table.sort(key=lambda x: x[1][1]) ## sorted freq of arr
        huffman_total_table.sort(key=lambda x: x[1][0], reverse=True) ## sorted len of code
        
        self.HuffmanTable = dict()
        code_len = huffman_total_table[0][1][1]
        code_val = 0b0
        for i in range(len(huffman_total_table)):
            key, (codeFreq, codeLen) = huffman_total_table[i]
            if codeLen != code_len:
                code_val <<= (codeLen-code_len)
                code_len = codeLen
            elif (i == len(huffman_total_table)-1) and code_len < 16:
                code_val = code_val << 1 if code_len < 16 else code_val+1
            self.HuffmanTable[key] = "{:0{}b}".format(code_val, codeLen)
            code_val += 1
        return self.HuffmanTable

    def __calc_compressed_size(self):
        compressed_size = 0
        for val, (freq, codeLen) in self.total_freq.items():
            compressed_size += freq * codeLen
        return compressed_size

    @classmethod
    def __calc_freq(self, arr):
        freq_dict = dict()
        for elem in arr:
            if elem in freq_dict:
                freq_dict[elem] += 1
            else:
                freq_dict[elem] = 1
        return freq_dict


