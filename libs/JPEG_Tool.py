import numpy as np
import cv2



fileDtype = np.int32
valueDtype = np.float64

## Color Space Converts
def cvtColor(img, type):
    if type=="RGB2YUV":
        return _rgb2ycbcr(img)
    elif type=="RGB2GRAY":
        return _rgb2ycbcr(img, onlyGray=True)
    elif type=="YUV2RGB":
        return _ycbcr2rgb(img)
    elif type=="GRAY2RGB":
        return _ycbcr2rgb(_gray2ycbcr(img))
    elif type=="GRAY2YUV":
        return _gray2ycbcr(img)
def img2ycbcr(img, color_space="RGB"):
    if color_space=="RGB":
        return _rgb2ycbcr(img)
    elif color_space=="GRAY":
        return _gray2ycbcr(img)

def _rgb2ycbcr(img, onlyGray=False):
    r,g,b=np.moveaxis(np.copy(img),-1,0)
    y=  ( 0.299*r+0.587*g+0.114*b) ## [0.~255.]
    cb= (-0.168*r-0.331*g+0.499*b)*(0 if onlyGray else 1) ## [-128.~127.]
    cr= ( 0.500*r-0.419*g-0.081*b)*(0 if onlyGray else 1) ## [-128.~127.]
    return np.stack((y,cb,cr),axis=2)

def _gray2ycbcr(img):
    return np.stack((img, np.zeros(img.shape), np.zeros(img.shape)),axis=2)

## DCT
### FDCT
def imageFDCT(img, block_size=8, color_space="YUV"):
    _img = np.copy(img)
    fdct = np.zeros((_img.shape), dtype=valueDtype)
    if color_space=="YUV":
        ## Prior to input to the Forward DCT (FDCT), the pixels are shifted about zero (-128 to +127).
        _img[:, :, 0] = _img[:, :, 0]-128.

    for i in range(0, _img.shape[0], block_size):
        for j in range(0, _img.shape[1], block_size):
            for k in range(_img.shape[2]):
                fdct[i:i+block_size,j:j+block_size, k] = cv2.dct(_img[i:i+block_size,j:j+block_size, k])
    return fdct

### IDCT
def imageIDCT(img, block_size=8, color_space="YUV"):
    _img = np.copy(img).astype(valueDtype)
    idct_img = np.zeros((img.shape), dtype=valueDtype)

    for i in range(0, idct_img.shape[0], block_size):
        for j in range(0, idct_img.shape[1], block_size):
            if img.ndim == 3:
                for k in range(idct_img.shape[2]):
                    idct_img[i:i+block_size,j:j+block_size, k] = cv2.idct(_img[i:i+block_size,j:j+block_size, k])
            else:
                idct_img[i:i+block_size,j:j+block_size] = cv2.idct(_img[i:i+block_size,j:j+block_size])
    if color_space=="YUV":
        if _img.ndim == 3:
            idct_img[:, :, 0] = idct_img[:, :, 0]+128.
        else:
            idct_img[:, :] = idct_img[:, :]+128.
    return idct_img

## Quantization
### Quantization Table
def QuantizationTable(quality=50):
    std_lumQT = np.array( ## Q_Y: 標準亮度量化表
        [[ 16,  11,  10,  16,  24,  40,  51,  61],
        [ 12,  12,  14,  19,  26,  58,  60,  55],
        [ 14,  13,  16,  24,  40,  57,  69,  56],
        [ 14,  17,  22,  29,  51,  87,  80,  62],
        [ 18,  22,  37,  56,  68, 109, 103,  77],
        [ 24,  35,  55,  64,  81, 104, 113,  92],
        [ 49,  64,  78,  87, 103, 121, 120, 101],
        [ 72,  92,  95,  98, 112, 100, 103,  99]], dtype=valueDtype)
    std_chrQT = np.array( ## Q_C: 標準色差量化表
        [[ 17,  18,  24,  47,  99,  99,  99,  99],
        [ 18,  21,  26,  66,  99,  99,  99,  99],
        [ 24,  26,  56,  99,  99,  99,  99,  99],
        [ 47,  66,  99,  99,  99,  99,  99,  99],
        [ 99,  99,  99,  99,  99,  99,  99,  99],
        [ 99,  99,  99,  99,  99,  99,  99,  99],
        [ 99,  99,  99,  99,  99,  99,  99,  99],
        [ 99,  99,  99,  99,  99,  99,  99,  99]], dtype=valueDtype)
            
    if(quality < 50):
        qualityScale = 5000 / quality
    else:
        qualityScale = 200 - quality * 2
            
    lumQT = np.array(np.floor((std_lumQT * qualityScale + 50) / 100))
    lumQT[lumQT == 0] = 1
    lumQT[lumQT > 255] = 255
    lumQT = lumQT.reshape([8, 8]).astype(fileDtype)
        
    chrQT = np.array(np.floor((std_chrQT * qualityScale + 50) / 100))
    chrQT[chrQT == 0] = 1
    chrQT[chrQT > 255] = 255
    chrQT = chrQT.reshape([8, 8]).astype(fileDtype)
            
    return lumQT,chrQT

## It's Qantized for YUV Color Space
### YUV's Color components example
### [
###     {'id': 1, "hscale": 1, "vscale": 1, "table_index": 0},
###     {'id': 2, "hscale": 1, "vscale": 1, "table_index": 1},
###     {'id': 3, "hscale": 1, "vscale": 1, "table_index": 1}
### ]
def imageQuantization(img, QuantizationTables, color_components):
    _img = np.zeros((img.shape), dtype=fileDtype)

    for k in range(img.shape[2]):
        _img[:, :, k] = np.round(
            img[:, :, k] / np.tile(QuantizationTables[color_components[k]["table_index"]], (img.shape[0]//8, img.shape[1]//8))
        )
    return _img

def imageDequantization(img, QuantizationTables, color_components):
    _img = np.zeros((img.shape), dtype=fileDtype)

    if img.ndim == 3:
        for k in range(img.shape[2]):
            _img[:, :, k] = np.round(
                img[:, :, k] * np.tile(QuantizationTables[color_components[k]["table_index"]], (img.shape[0]//8, img.shape[1]//8))
            )
    else:
        _img[:, :] = np.round(
             img[:, :] * np.tile(QuantizationTables[color_components[0]["table_index"]], (img.shape[0]//8, img.shape[1]//8))
        )
    
    return _img

## ZigZag
### input:  H*W*Comp*block(shape(8, 8)) size color space array
### output: H*W*Comp*zigzag(shape(64, )) size array
def imageZigzag(img, block_size=8):
    zigzag = []
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            comp = []
            for k in range(img.shape[2]):
                comp.append(_zigzag(img[i:i+block_size,j:j+block_size, k]))
            zigzag.append(comp)
    return np.array(zigzag)

### input: nDim array
### output: zigzag(shape()) array
def _zigzag(block, block_size=8):
    index_list = _zigzag_indexList(block_size)
    return np.array([np.array(block).flatten()])[:, index_list].flatten()

def _dezigzag_indexList(block, block_size=8):
    Zigzag_index_list = _zigzag_indexList(block_size)
    if Zigzag_index_list.ndim == 2:
        _1Dzig = Zigzag_index_list[:, 0]*block_size + Zigzag_index_list[:, 1]
    else:
        _1Dzig = Zigzag_index_list
    Dezigzag_index_list = sorted(range(len(_1Dzig)), key=lambda k: _1Dzig[k])
    return np.array(Dezigzag_index_list)

def _zigzag_indexList(n):
    index_list = np.array([])
    for i in range(n):
        indexs = np.zeros((i+1, 2), dtype=np.uint8)
        indexs[:, 0] = np.arange(i+1) if i % 2 else np.arange(i+1)[::-1]
        indexs[:, 1] = indexs[::-1, 0]
        try:
            index_list = np.concatenate((index_list, indexs))
        except:
            index_list = indexs
    for i in list(reversed(range(n-1))):
        indexs = np.zeros((i+1, 2), dtype=np.uint8)
        indexs[:, 1] = (np.arange(i+1)+n-1-i) if i % 2 == 0 else (np.arange(i+1)+n-1-i)[::-1]
        indexs[:, 0] = indexs[::-1, 1]
        index_list = np.concatenate((index_list, indexs))
    
    return index_list[:, 0] * n + index_list[:, 1]

## block MCU
### input: H*W*Comp*zigzag(shape(64, )) size array
### outpu: H*W*Comp*MCU_Struct("DPCM", "RLE") size array
def MCU(zigzaged_array):
    last_DC = np.zeros(zigzaged_array.shape[1])
    MCU_block_list = []
    for zigzag_block in zigzaged_array:
        MCU_comp = []
        for comp_index in range(len(zigzag_block)):
            MCU_comp.append(getMCUfromBlock(zigzag_block[comp_index], last_DC[comp_index]))
            last_DC[comp_index] = zigzag_block[comp_index][0]
        MCU_block_list.append(MCU_comp)
    
    return MCU_block_list

### input: components block, and last DC value
### outpu: H*W*Comp*MCU_Struct("DPCM", "RLE") size array
def getMCUfromBlock(block, last_DC):
    DPCM = int(block[0]-last_DC)
    RLE = []
    unZeroIndex_list = np.argwhere(block[1:] != 0).flatten()
    last_unZero_index = 0
    for i in unZeroIndex_list:
        while(i+1 - last_unZero_index > 16):
            RLE.append((15, 0))
            last_unZero_index += 16
        RLE.append((i-last_unZero_index, block[i+1]))
        last_unZero_index+=1
    if last_unZero_index < 63:
        RLE.append((0, 0))
    return {"DPCM": DPCM, "RLE": np.array(RLE, dtype=np.int32)}

def _HuffmanEncoding(self, DEBUG=False):
    mcus_encoded = []
    i = 0
    for mcu in self.MCUs:
        i+=1
        if DEBUG: print("## MCU {}".format(i))
        mcus_encoded.append(self.HuffmanEncodingMCU(mcu, DEBUG))
    if DEBUG:
        print("## mcus_encoded")
        pprint(mcus_encoded)
    self.encodedMCUs = "".join(mcus_encoded)
    return self.encodedMCUs