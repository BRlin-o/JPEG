import numpy as np

class Scheme_Huang():
    def __init__(self, block):
        self.block = np.array(block)
        self.rle = self.block[1:64]
        self.dpcm = self.block[0]

    def __initPrepare__(self, block):
        self.__init__(block)
        self.pre_Embedding()

    def getBlock(self):
        block = np.copy(self.block)
        block[0] = self.dpcm
        block[1:64] = self.rle
        return block

    def pre_Embedding(self):
        self.inner_index = np.argwhere((self.rle == 1) | (self.rle == -1))
        self.rle = self.rle + (self.rle > 1) - (self.rle < -1)
        self.block[1:64] = self.rle

    def freeSize(self):
        return self.inner_index.shape[0]

    def embedding(self, data):
        if len(data) > self.freeSize():
            raise Exception("The data size is bigger than the free size")

        # self.pre_Embedding()
        rle = np.copy(self.rle)
        for i in range(len(data)):
            # print(i, self.block[self.inner_index[i]])
            # self.block[self.inner_index[i]] = self.embed(self.block[self.inner_index[i]], int(data[i]))
            rle[self.inner_index[i]] = self.embed(self.rle[self.inner_index[i]], int(data[i]))

        self.block[1:64] = self.rle = rle
        return self.block

    def reStoring(self):
        rle = np.copy(self.rle).flatten()
        message = []
        for index in range(len(rle)):
            m = self.reSign(rle[index])
            rs = self.reStore(rle[index])
            rle[index] = rle[index] - (rs if m is None else rs*m)
            if m is not None:
                message.append(m)
        self.block[1:64] = self.rle = rle
        return self.block.reshape(self.block.shape), message

    @classmethod
    def sign(self, c_i):
        if c_i > 0:
            return 1
        elif c_i == 0:
            return 0
        elif c_i < 0:
            return -1
    @classmethod
    def reSign(self, c_i_p):
        if abs(c_i_p) == 1:
            return 0
        elif abs(c_i_p) == 2:
            return 1
        else:
            return None
    @classmethod
    def embed(self, c_i, s):
        if abs(c_i) == 1:
            return c_i + self.sign(c_i)*s
        elif abs(c_i) > 1:
            return c_i + self.sign(c_i)
    @classmethod
    def reStore(self, c_i_p):
        if abs(c_i_p) >= 1 & abs(c_i_p) <= 2:
            return self.sign(c_i_p)
        elif abs(c_i_p) >= 3:
            return c_i_p-self.sign(c_i_p)
        else:
            return 0

class Scheme_Extended():
    def __init__(self, block):
        self.block = np.array(block)
        self.dpcm = self.block[0]
        self.rle = self.block[1:64]

    def __initPrepare__(self, block):
        self.__init__(block)
        self.pre_Embedding()

    def getBlock(self):
        block = np.copy(self.block)
        block[0] = self.dpcm
        block[1:64] = self.rle
        return block

    def pre_Embedding(self):
        if np.any((self.rle == -3) | (self.rle == 3) | (self.rle == 4) | (self.rle == -4)):
            raise Exception("block contains -3, 3, 4, -4")
        self.inner_index = np.argwhere((self.rle == 1) | (self.rle == -1) | (self.rle == 2) | (self.rle == -2))

    def freeSize(self):
        return self.inner_index.shape[0]

    def embedding(self, data):
        if len(data) > self.freeSize():
            raise Exception("data size is not equal to free size")

        # self.pre_Embedding()
        rle = np.copy(self.rle)
        for i in range(len(data)):
            try:
                rle[self.inner_index[i]] = self.embed(self.rle[self.inner_index[i]], int(data[i]))
            except:
                print("self.inner_index[i]", self.inner_index[i])
                print("self.block[self.inner_index[i]]", self.rle[self.inner_index[i]])
                print("data[i]", data[i])
                raise Exception("error")
        # self.embedded_block = new_block
        self.block[1:64] = self.rle = rle
        # return self.embedded_block
        return self.block

    def reStoring(self):
        rle = np.copy(self.rle).flatten()
        message = []
        for index in range(len(rle)):
            if (abs(rle[index]) >= 1) & (abs(rle[index]) <= 4):
                m = self.reSign(rle[index])
                rs = self.reStore(rle[index])
                if m is not None:
                    rle[index] = rle[index] - rs
                    message.append(m)
        self.block[1:64] = self.rle = rle
        return self.block.reshape(self.block.shape), message
    @classmethod
    def sign(self, c_i):
        if c_i > 0:
            return 1
        elif c_i == 0:
            return 0
        elif c_i < 0:
            return -1
    @classmethod
    def reSign(self, c_i_p):
        if (abs(c_i_p) == 1) | (abs(c_i_p) == 3):
            return 0
        elif (abs(c_i_p) == 2) | (abs(c_i_p) == 4):
            return 1
        else:
            return None
    @classmethod
    def embed(self, c_i, s):
        if abs(c_i) == 1:
            return c_i + self.sign(c_i)*s
        elif abs(c_i) == 2:
            return c_i + self.sign(c_i)*(s+1)
        else:
            return c_i
    
    @classmethod
    def reStore(self, c_i_p):
        # if (abs(c_i_p) >= 1) & (abs(c_i_p) <= 2):
        #     return self.sign(c_i_p)
        # elif (abs(c_i_p) >= 3) & (abs(c_i_p) <= 4):
        #     return (self.sign(c_i_p)+(1 if c_i_p > 0 else -1))
        # else:
        #     return 0
        if (abs(c_i_p) == 2) | (abs(c_i_p) == 3):
            return self.sign(c_i_p)
        elif (abs(c_i_p) == 4):
            return 2*self.sign(c_i_p)
        else:
            return 0