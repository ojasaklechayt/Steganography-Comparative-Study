import os
import xlwt
import shutil
import cv2
import sys
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy import signal
import random
import base64
from skimage.metrics import structural_similarity as compare_ssim
import xlsxwriter
import matplotlib.patches as mpatches
import codecs

quant = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

class RPEEncryptDecrypt:
    @staticmethod
    def text_to_binary(message):
        return ''.join(format(ord(char), '08b') for char in message)

    @staticmethod
    def binary_to_text(binary_str):
        return ''.join([chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)])

    @staticmethod
    def encode_rpe(original_image, secret_message):
        secret_message += 'H'  # Choose a character that is unlikely to appear in the message
        binary_message = RPEEncryptDecrypt.text_to_binary(secret_message)

        if len(binary_message) > original_image.size[0] * original_image.size[1] * 3:
            raise ValueError("Message is too long to be encoded in the image.")

        pixels = list(original_image.getdata())
        random.seed(42)  # Setting a seed for consistency

        for i in range(len(binary_message)):
            pixel_index = random.randint(0, len(pixels) - 1)
            pixel_value = list(pixels[pixel_index])
            channel_index = random.randint(0, 2)  # Randomly select a channel (R, G, or B)
            pixel_value[channel_index] &= 254  # Clear the least significant bit
            pixel_value[channel_index] |= int(binary_message[i])  # Set the least significant bit
            pixels[pixel_index] = tuple(pixel_value)

        new_img = Image.new(original_image.mode, original_image.size)
        new_img.putdata(pixels)
        return new_img

    @staticmethod
    def decode_rpe(encoded_image):
        pixels = list(encoded_image.getdata())
        random.seed(42)  # Setting the same seed for consistency

        binary_message = ''
        message = ''
        for i in range(len(pixels)):
            pixel_index = random.randint(0, len(pixels) - 1)
            pixel_value = list(pixels[pixel_index])
            channel_index = random.randint(0, 2)  # Randomly select a channel (R, G, or B)
            binary_message += str(pixel_value[channel_index] & 1)  # Extracting the least significant bit

            # Every 8 bits, convert to a character and check for the termination character
            if len(binary_message) >= 8:
                char = RPEEncryptDecrypt.binary_to_text(binary_message[:8])
                if char == 'H':  # Termination character
                    break
                message += char
                binary_message = binary_message[8:]

        return message


class DWT:
    def encode_image(self, img_data, secret_msg):
        # Convert the input image data to a NumPy array
        img_arr = np.array(img_data)

        # Convert the input image to RGB if it's not already
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

        height, width, _ = img_arr.shape
        length = len(secret_msg)
        index = 0

        # Encode the message into the image data
        for row in range(height):
            for col in range(width):
                if index < length:
                    c = secret_msg[index]
                    asc = ord(c)
                else:
                    asc = img_arr[row, col, 0]  # Assuming the message is encoded in the blue channel
                img_arr[row, col, 0] = asc
                index += 1

        # Convert the NumPy array back to a PIL image
        encoded_pil_img = Image.fromarray(img_arr)

        return encoded_pil_img

    def decode_image(self, img_data, length):
        # Convert the input image data to a NumPy array
        img_arr = np.array(img_data)

        # Convert the input image to RGB if it's not already
        if len(img_arr.shape) == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

        msg = ""
        height, width, _ = img_arr.shape
        index = 0

        # Decode the hidden message from the image data
        for row in range(height):
            for col in range(width):
                if index < length:
                    pixel_value = img_arr[row, col, 0]  # Assuming the message is encoded in the blue channel
                    msg += chr(pixel_value)
                    index += 1

        return msg


class DCT():
    def __init__(self):
        self.message = None
        self.bitMess = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0

    def encode_image(self, img, secret_msg):
        row, col = img.shape[:2]
        self.message = str(len(secret_msg)) + '*' + secret_msg
        self.bitMess = self.toBits()
        self.oriRow, self.oriCol = row, col
        if ((col / 8) * (row / 8) < len(secret_msg)):
            print("Error: Message too large to encode in image")
            return False
        if row % 8 != 0 or col % 8 != 0:
            img = self.addPadd(img, row, col)
        row, col = img.shape[:2]
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        imgBlocks = [np.round(bImg[j:j+8, i:i+8]-128) for (j, i) in itertools.product(range(0, row, 8),
                                                                       range(0, col, 8))]
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        quantizedDCT = [np.round(dct_Block / quant) for dct_Block in dctBlocks]
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            DC[7] = self.bitMess[messIndex][letterIndex]
            DC = np.packbits(DC)
            DC = np.float32(DC)
            DC = DC - 255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex + 1
            if letterIndex == 8:
                letterIndex = 0
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return sImg

    def decode_image(self, img):
        row, col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        bImg, gImg, rImg = cv2.split(img)
        bImg = np.float32(bImg)
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j, i) in itertools.product(range(0, row, 8),
                                                                              range(0, col, 8))]
        quantizedDCT = [img_Block / quant for img_Block in imgBlocks]
        i = 0
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff += (0 & 1) << (7 - i)
            elif DC[7] == 0:
                buff += (1 & 1) << (7 - i)
            i = 1 + i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i = 0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize)) + 1:]
        sImgBlocks = [quantizedBlock * quant + 128 for quantizedBlock in quantizedDCT]
        sImg = []
        for chunkRowBlocks in self.chunks(sImgBlocks, col / 8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg, gImg, rImg))
        return ''

    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]

    def addPadd(self, img, row, col):
        img = cv2.resize(img, (col + (8 - col % 8), row + (8 - row % 8)))
        return img

    def toBits(self):
        bits = []
        for char in self.message:
            binval = bin(ord(char))[2:].rjust(8, '0')
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8, '0')
        return bits


class LSB():
    #encoding part :
    def encode_image(self,img, msg):
        length = len(msg)
        if length > 255:
            print("text too long! (don't exeed 255 characters)")
            return False
        encoded = img.copy()
        width, height = img.size
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))
                # first value is length of msg
                if row == 0 and col == 0 and index < length:
                    asc = length
                elif index <= length:
                    c = msg[index -1]
                    asc = ord(c)
                else:
                    asc = b
                encoded.putpixel((col, row), (r, g , asc))
                index += 1
        return encoded
    
    #decoding part :
    def decode_image(self,img):
        width, height = img.size
        msg = ""
        index = 0
        for row in range(height):
            for col in range(width):
                if img.mode != 'RGB':
                    r, g, b ,a = img.getpixel((col, row))
                elif img.mode == 'RGB':
                    r, g, b = img.getpixel((col, row))  
                # first pixel r value is length of message
                if row == 0 and col == 0:
                    length = b
                elif index <= length:
                    msg += chr(b)
                index += 1
        lsb_decoded_image_file = "lsb_" + original_image_file
        #img.save(lsb_decoded_image_file)
        ##print("Decoded image was saved!")
        return msg

class SpreadSpectrumSteganography:
    def __init__(self, strength=1):
        self.strength = strength
        self.pseudo_random_seq = None

    def encrypt(self, cover_image, secret_message):
        # Convert secret message to binary
        secret_message_binary = ''.join(format(ord(char), '08b') for char in secret_message)

        # Convert the cover image to numpy array
        cover_image_array = np.array(cover_image)

        # Get the dimensions of the cover image
        height, width, channels = cover_image_array.shape

        # Calculate the number of bits we can encode
        max_bits_to_encode = height * width * channels * self.strength

        if len(secret_message_binary) > max_bits_to_encode:
            raise ValueError("Message too large to be encoded in the given image with the specified strength")

        # Generate pseudo-random sequence based on image shape
        self.pseudo_random_seq = np.random.randint(0, 2, size=cover_image_array.shape)

        # Spread spectrum encryption
        encrypted_image = cover_image_array.copy()
        idx = 0
        for row in range(encrypted_image.shape[0]):
            for col in range(encrypted_image.shape[1]):
                for channel in range(encrypted_image.shape[2]):
                    if idx < len(secret_message_binary):
                        # Apply spread spectrum encoding
                        encrypted_image[row, col, channel] = (encrypted_image[row, col, channel] + self.strength * (-1) ** int(secret_message_binary[idx]) * self.pseudo_random_seq[row, col, channel]) % 256
                        idx += 1
                    else:
                        break

        return encrypted_image

    def decrypt(self, encrypted_image):
        encrypted_image_array = np.array(encrypted_image)

        # Flatten the image array
        flattened_image = encrypted_image_array.reshape(-1, encrypted_image_array.shape[-1])

        # Flatten the pseudo-random sequence
        flattened_seq = self.pseudo_random_seq.reshape(flattened_image.shape)

        # Retrieve the encoded bits and convert them to binary string
        encoded_bits = ((flattened_image - flattened_image % self.strength) // self.strength).astype(int) ^ flattened_seq
        binary_msg = ''.join(str(bit) for bit in encoded_bits.flatten())

        binary_msg = ''.join(char for char in binary_msg if char in '01')
        
        # Convert binary string to text
        decoded_msg = ''.join(chr(int(binary_msg[i:i+8], 2)) for i in range(0, len(binary_msg), 8))

        return decoded_msg
    
    def generate_pseudo_random_sequence(self, img_shape):
        np.random.seed(123)
        self.pseudo_random_seq = np.random.randint(0, 2, size=img_shape)


class Compare:
    @staticmethod
    def ssim(img1, img2):
        if len(img1.shape) > 2:
            img1 = np.mean(img1, axis=2)
        if len(img2.shape) > 2:
            img2 = np.mean(img2, axis=2)
        
        return compare_ssim(img1, img2, data_range=img1.max() - img1.min())
    
    @staticmethod
    def correlation(img1, img2):
        if len(img1.shape) > 2:
            img1 = np.mean(img1, axis=2)
        if len(img2.shape) > 2:
            img2 = np.mean(img2, axis=2)
        
        return signal.correlate2d(img1, img2, mode='valid')

    @staticmethod
    def meanSquareError(img1, img2):
        error = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
        error /= float(img1.shape[0] * img1.shape[1])
        return error

    @staticmethod
    def psnr(img1, img2):
        mse = Compare.meanSquareError(img1, img2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    @staticmethod
    def embedding_capacity(img_size):
        # Number of bits that can be embedded per pixel
        return img_size * 3 * 8

def generate_comparison_chart(data, output_filename):
    methods = [str(method) if method is not None else 'Unknown' for method, *_ in data]
    ssim_values = [float(ssim) if ssim is not None else 0.0 for _, ssim, *_ in data]
    correlation_values = [float(correlation) if correlation is not None else 0.0 for _, _, correlation, *_ in data]
    psnr_values = [float(psnr) if psnr is not None else 0.0 for _, _, _, psnr, *_ in data]
    capacity_values = [float(capacity) if capacity is not None else 0.0 for _, _, _, _, capacity, *_ in data]
    mse_values = [float(mse) if mse is not None else 0.0 for _, _, _, _, _, mse in data]

    plt.figure(figsize=(12, 8))

    # Plot SSIM
    plt.subplot(2, 3, 1)
    for method, ssim in zip(methods, ssim_values):
        plt.bar(method, ssim, label=method)
    plt.title('SSIM')
    plt.xticks(rotation=45)

    # Plot Correlation
    plt.subplot(2, 3, 2)
    for method, correlation in zip(methods, correlation_values):
        plt.bar(method, correlation, label=method)
    plt.title('Correlation')
    plt.xticks(rotation=45)

    # Plot PSNR
    plt.subplot(2, 3, 3)
    for method, psnr in zip(methods, psnr_values):
        plt.bar(method, psnr, label=method)
    plt.title('PSNR')
    plt.xticks(rotation=45)

    # Plot Capacity
    plt.subplot(2, 3, 4)
    for method, capacity in zip(methods, capacity_values):
        plt.bar(method, capacity, label=method)
    plt.title('Embedding Capacity')
    plt.xticks(rotation=45)

    # Plot MSE
    plt.subplot(2, 3, 5)
    for method, mse in zip(methods, mse_values):
        plt.bar(method, mse, label=method)
    plt.title('MSE')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    if os.path.exists("Encoded_image/"):
        shutil.rmtree("Encoded_image/")
    if os.path.exists("Decoded_output/"):
        shutil.rmtree("Decoded_output/")
    if os.path.exists("Comparison_result/"):
        shutil.rmtree("Comparison_result/")
    os.makedirs("Encoded_image/")
    os.makedirs("Decoded_output/")
    os.makedirs("Comparison_result/")

    original_image_file = ""
    lsb_encoded_image_file = ""
    dct_encoded_image_file = ""
    dwt_encoded_image_file = ""
    spread_spectrum_encoded_image_file = ""
    rpe_encoded_image_file = ""  # Corrected initialization

    while True:
        m = input("To encode press '1', to decode press '2', to compare press '3', press any other button to close: ")

        if m == "1":
            os.chdir("Original_image/")
            original_image_file = input("Enter the name of the file with extension : ")
            img = Image.open(original_image_file)
            lsb_img = Image.open(original_image_file)
            dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(original_image_file)
            print("Description : ", lsb_img, "\nMode : ", lsb_img.mode)
            secret_msg = input("Enter the message you want to hide: ")
            print("The message length is: ", len(secret_msg))
            os.chdir("..")
            os.chdir("Encoded_image/")
            # Encoding using different methods
            lsb_img_encoded = LSB().encode_image(lsb_img, secret_msg)
            dct_img_encoded = DCT().encode_image(dct_img, secret_msg)
            dwt_img_encoded = DWT().encode_image(np.array(dwt_img), secret_msg)
            # Integration of Spread Spectrum
            spread_spectrum_steganography = SpreadSpectrumSteganography()
            encrypted_image = spread_spectrum_steganography.encrypt(spread_spectrum_img, secret_msg)
            rpe_img_encoded_array = RPEEncryptDecrypt().encode_rpe(img, secret_msg)


            lsb_encoded_image_file = "lsb_" + original_image_file
            lsb_img_encoded.save(lsb_encoded_image_file)
            dct_encoded_image_file = "dct_" + original_image_file
            cv2.imwrite(dct_encoded_image_file, dct_img_encoded)
            dwt_encoded_image_file = "dwt_" + original_image_file
            dwt_img_encoded_array = np.array(dwt_img_encoded)
            cv2.imwrite(dwt_encoded_image_file, dwt_img_encoded_array)
            spread_spectrum_encoded_image_file = "spread_spectrum_" + original_image_file
            Image.fromarray(encrypted_image).save(spread_spectrum_encoded_image_file)
            rpe_encoded_image_file = "rpe_" + original_image_file
            Image.fromarray(np.uint8(rpe_img_encoded_array)).save(rpe_encoded_image_file)

            print("Encoded images were saved!")
            os.chdir("..")

        # Decoding Section
        elif m == "2":
            os.chdir("Encoded_image/")
            # Load encoded images
            lsb_img = Image.open(lsb_encoded_image_file)
            dct_img = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(spread_spectrum_encoded_image_file)
            rpe_img = Image.open(rpe_encoded_image_file)
            os.chdir("..")
            os.makedirs("Decoded_output/", exist_ok=True)
            os.chdir("Decoded_output/")

            # Decoding using different methods
            lsb_hidden_text = LSB().decode_image(lsb_img)
            dct_hidden_text = DCT().decode_image(dct_img)
            dwt_hidden_text = DWT().decode_image(np.array(dwt_img), len(secret_msg))
            spread_spectrum_steganography = SpreadSpectrumSteganography()
            spread_spectrum_steganography.generate_pseudo_random_sequence(np.array(encrypted_image).shape)
            encrypted_image_array = np.array(spread_spectrum_img)
            decrypted_message = spread_spectrum_steganography.decrypt(encrypted_image_array)
            rpe_hidden_text = RPEEncryptDecrypt().decode_rpe(rpe_img)

            # Function for safe writing with proper encoding
            def safe_write(file, text):
                try:
                    file.write(text)
                except UnicodeEncodeError:
                    # Handle non-ASCII characters by ignoring them
                    file.write(text.encode('ascii', 'ignore').decode())

            # Save decoded messages to text files with proper encoding and error handling
            with open("lsb_decoded.txt", "w", encoding="utf-8") as lsb_file:
                safe_write(lsb_file, lsb_hidden_text)
            with open("dct_decoded.txt", "w", encoding="utf-8") as dct_file:
                safe_write(dct_file, dct_hidden_text)
            with open("dwt_decoded.txt", "w", encoding="utf-8") as dwt_file:
                safe_write(dwt_file, dwt_hidden_text)
            with open("spread_spectrum_decoded.txt", "w", encoding="latin-1") as spread_spectrum_file:
                safe_write(spread_spectrum_file, decrypted_message)
            with open("rpe_decoded.txt", "w", encoding="latin-1") as rpe_file:
                safe_write(rpe_file, rpe_hidden_text)

            print("Decoded messages were saved in the Decoded_output folder.")
            os.chdir("..")

        elif m == "3":
            # Comparison Section
            os.chdir("Original_image/")
            original_img = Image.open(original_image_file)
            lsb_img = Image.open(original_image_file)
            dct_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            dwt_img = cv2.imread(original_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_img = Image.open(original_image_file)
            rpe_img = Image.open(original_image_file)
            os.chdir("..")
            os.chdir("Encoded_image/")
            lsb_encoded_img = Image.open(lsb_encoded_image_file)
            dct_encoded_img = cv2.imread(dct_encoded_image_file, cv2.IMREAD_UNCHANGED)
            dwt_encoded_img = cv2.imread(dwt_encoded_image_file, cv2.IMREAD_UNCHANGED)
            spread_spectrum_encoded_img = Image.open(spread_spectrum_encoded_image_file)
            rpe_encoded_img = Image.open(rpe_encoded_image_file)
            os.chdir("..")
            os.makedirs("Comparison_result/", exist_ok=True)
            os.chdir("Comparison_result/")

            # Convert images to numpy arrays
            original_img_array = np.array(original_img)
            lsb_encoded_img_array = np.array(lsb_encoded_img)
            dct_encoded_img_array = dct_encoded_img
            dwt_encoded_img_array = dwt_encoded_img
            spread_spectrum_encoded_img_array = np.array(spread_spectrum_encoded_img)
            rpe_encoded_img_array = np.array(rpe_encoded_img)

            # Calculate SSIM, correlation, PSNR, embedding capacity, and MSE
            ssim_lsb = Compare.ssim(original_img_array, lsb_encoded_img_array)
            ssim_dct = Compare.ssim(original_img_array, dct_encoded_img_array)
            ssim_dwt = Compare.ssim(original_img_array, dwt_encoded_img_array)
            ssim_spread_spectrum = Compare.ssim(original_img_array, spread_spectrum_encoded_img_array)
            ssim_rpe = Compare.ssim(original_img_array, rpe_encoded_img_array)

            correlation_lsb = Compare.correlation(original_img_array, lsb_encoded_img_array)
            correlation_dct = Compare.correlation(original_img_array, dct_encoded_img_array)
            correlation_dwt = Compare.correlation(original_img_array, dwt_encoded_img_array)
            correlation_spread_spectrum = Compare.correlation(original_img_array, spread_spectrum_encoded_img_array)
            correlation_rpe = Compare.correlation(original_img_array, rpe_encoded_img_array)

            psnr_lsb = Compare.psnr(original_img_array, lsb_encoded_img_array)
            psnr_dct = Compare.psnr(original_img_array, dct_encoded_img_array)
            psnr_dwt = Compare.psnr(original_img_array, dwt_encoded_img_array)
            psnr_spread_spectrum = Compare.psnr(original_img_array, spread_spectrum_encoded_img_array)
            psnr_rpe = Compare.psnr(original_img_array, rpe_encoded_img_array)

            capacity_lsb = Compare.embedding_capacity(lsb_encoded_img_array.size)
            capacity_dct = Compare.embedding_capacity(dct_encoded_img_array.size)
            capacity_dwt = Compare.embedding_capacity(dwt_encoded_img_array.size)
            capacity_spread_spectrum = Compare.embedding_capacity(spread_spectrum_encoded_img_array.size)
            capacity_rpe = Compare.embedding_capacity(rpe_encoded_img_array.size)

            # Calculate MSE
            mse_lsb = np.mean((original_img_array - lsb_encoded_img_array) ** 2)
            mse_dct = np.mean((original_img_array - dct_encoded_img_array) ** 2)
            mse_dwt = np.mean((original_img_array - dwt_encoded_img_array) ** 2)
            mse_spread_spectrum = np.mean((original_img_array - spread_spectrum_encoded_img_array) ** 2)
            mse_rpe = np.mean((original_img_array - rpe_encoded_img_array) ** 2)

            # Create a workbook and add a worksheet
            workbook = xlsxwriter.Workbook("comparison_result.xlsx")
            worksheet = workbook.add_worksheet()

            # Write column headers
            headers = ["Method", "SSIM", "Correlation", "PSNR", "Embedding Capacity", "MSE"]
            for col, header in enumerate(headers):
                worksheet.write(0, col, header)

            # Write data rows
            data = [
                ["LSB", ssim_lsb, correlation_lsb, psnr_lsb, capacity_lsb, mse_lsb],
                ["DCT", ssim_dct, correlation_dct, psnr_dct, capacity_dct, mse_dct],
                ["DWT", ssim_dwt, correlation_dwt, psnr_dwt, capacity_dwt, mse_dwt],
                ["Spread Spectrum", ssim_spread_spectrum, correlation_spread_spectrum, psnr_spread_spectrum, capacity_spread_spectrum, mse_spread_spectrum],
                ["RPE", ssim_rpe, correlation_rpe, psnr_rpe, capacity_rpe, mse_rpe]
            ]
            for row, row_data in enumerate(data, start=1):
                for col, cell_data in enumerate(row_data):
                    worksheet.write(row, col, cell_data)

            comparison_data = [
                ["LSB", ssim_lsb, correlation_lsb, psnr_lsb, capacity_lsb, mse_lsb],
                ["DCT", ssim_dct, correlation_dct, psnr_dct, capacity_dct, mse_dct],
                ["DWT", ssim_dwt, correlation_dwt, psnr_dwt, capacity_dwt, mse_dwt],
                ["Spread Spectrum", ssim_spread_spectrum, correlation_spread_spectrum, psnr_spread_spectrum, capacity_spread_spectrum, mse_spread_spectrum],
                ["RPE", ssim_rpe, correlation_rpe, psnr_rpe, capacity_rpe, mse_rpe]
            ]
            
            generate_comparison_chart(comparison_data, "comparison_chart.png")
        
            # Close the workbook
            workbook.close()

            print("Comparison result saved in comparison_result.xlsx")

            os.chdir("..")

        else:
            print("Exiting...")
            break