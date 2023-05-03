import cv2
import numpy as np


def deteksi_garis():
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)
    cv2.imshow('Original Image', img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Deteksi Garis', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Tidak dapat mendeteksi garis pada gambar')


def deteksi_tepi():
    img = cv2.imread('image.jpg', 0)

    # Operator Roberts 
    edges_roberts = cv2.filter2D(img, -1, np.array([[0, 1], [-1, 0]]))

    # Operator Prewitt 
    edges_prewitt = cv2.filter2D(img, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

    # Operator Sobel 
    edges_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    cv2.imshow('Original Image', img)
    cv2.imshow('Operator Roberts', edges_roberts)
    cv2.imshow('Operator Prewitt', edges_prewitt)
    cv2.imshow('Operator Sobel', edges_sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_dwi_aras():
    img = cv2.imread('image.jpg', 0)

    # Histogram equalization
    equ = cv2.equalizeHist(img)

    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Dwi Aras', equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_global():
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Global', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_lokal():
    img = cv2.imread('image.jpg')
    block_size = 11
    C = 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Lokal', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_metode_otsu():
    img = cv2.imread('image.jpg', 0)

    # Otsu thresholding
    ret, thresh = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Metode Otsu', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_adaptif():
    img = cv2.imread('image.jpg', 0)

    # Pengambangan Adapmean
    adapmean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    # Pengambangan Adapmedian
    adapmedian = cv2.medianBlur(img, 3)
    adapmedian = cv2.adaptiveThreshold(adapmedian, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    # Pengambangan Adapmaxmin
    adapmaxmin = cv2.blur(img, (5,5))
    adapmaxmin = cv2.adaptiveThreshold(adapmaxmin, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Adapmean', adapmean)
    cv2.imshow('Pengambangan Adapmedian', adapmedian)
    cv2.imshow('Pengambangan Adapmaxmin', adapmaxmin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pengambangan_berdasarkan_entropi():
    img = cv2.imread('image.jpg', 0)

    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adapthist = clahe.apply(img)

    cv2.imshow('Original Image', img)
    cv2.imshow('Pengambangan Berdasarkan Entropi', adapthist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentasi_warna():
    img = cv2.imread('image.jpg')
    k = 3

    # Ubah citra menjadi 2D array
    pixel_values = img.reshape((-1, 3))

    # Konversi tipe data menjadi float32
    pixel_values = np.float32(pixel_values)

    # Terapkan algoritma K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Ubah nilai piksel menjadi nilai pusat yang sesuai
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # Ubah kembali bentuk citra
    segmented_image = segmented_data.reshape((img.shape))

    # Tampilkan citra hasil segmentasi
    cv2.imshow('Original Image', img)
    cv2.imshow('Segmentasi Warna', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """ Pemanggilan semua metode """
    deteksi_garis()
    # deteksi_tepi()
    # pengambangan_dwi_aras()
    # pengambangan_global()
    # pengambangan_lokal()
    # pengambangan_metode_otsu()
    # pengambangan_adaptif()
    # pengambangan_berdasarkan_entropi()
    # segmentasi_warna()


if __name__ == "__main__":
    main()
