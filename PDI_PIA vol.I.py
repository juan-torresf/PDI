import os, sys
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import pydicom
import kagglehub
import tkinter as tk
from tkinter import filedialog, filedialog, Scale, LabelFrame, Canvas, Button, Frame
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from webbrowser import open

# INTEGRANTES DEL EQUIPO 4 -------------------------------------------------------------------
#Benito Woo Zozaya 2177866
#Karen Melissa Morales Moya 2043909
#Carlos Arturo Arguelles Segura 2096281
#Juan Gerardo Torres Flores 2074382

#VARIABLES INICIALIZADAS
filename = None                 
img = None                      
img_original = None             
img_mejor = None                
img_gauss = None                
mostrar = None                  
mostrar_editada = None          
img_antes_de_segmentar = None
invertir = False


# IMAGENES KAGGLE-----------------------------------------------------------------------------
path = kagglehub.dataset_download("benitowoo/pancreas-and-project-images")
print("Path to dataset files:", path)


#CONFIGURACIÓN GENERAL DE LA PÁGINA ----------------------------------------------------------
root = tk.Tk()
root.title("Software de Procesamiento de Imágenes Médicas - Equipo 4")
root.configure(bg="#082338")
root.state('zoomed')
#root.attributes('-fullscreen', True)
root.bind("<Escape>", lambda e: root.destroy())

message = tk.Label(root, text="PIA EQUIPO4", bg = "#082338", font=("Papyrus", 25), fg = "#FFFFFF")
message.pack()

#CANVA PARA IMAGEN ORIGINAL
canva_img_original = Canvas(root, bg = "#FF0000",bd = 5)
canva_img_original.place(relx=0.95, rely=0.07, anchor="ne", relwidth=0.42, relheight=0.7)

#CANVA PARA IMAGEN EDITADA
canva_img_editada = Canvas(root, bg = "#FF0000",bd = 5)
canva_img_editada.place(relx=0.05, rely=0.07, anchor="nw", relwidth=0.42, relheight=0.7)

#FRAME PARA BOTONES
button_frame = Frame(root, bg="#43CD31", bd=5)
button_frame.place(relx=0.5, rely=0.9, anchor="center", relwidth=.9)

button_frame_top = Frame(button_frame, bg="#43CD31", bd=5)
button_frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
button_frame_bottom = Frame(button_frame, bg="#43CD31", bd=5)
button_frame_bottom.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

rocky_place = Frame(root, width=75, height=75, bg="#082338")
rocky_place.pack(side="left", anchor="nw")
image_path= os.path.join(path, 'AMAZE.png')
rocky = Image.open(image_path)
rocky_image = rocky.resize((75, 75), Image.Resampling.LANCZOS)
image = ImageTk.PhotoImage(rocky_image)
#PARA YO: MODIFICA LA UBICACIÓN DEL ROCKY PARA QUE SI SE VEA :(
amaze = tk.Label(rocky_place,image=image,bg = "#082338")
amaze.pack(expand=True, fill="both")

# INSERTAR IMAGEN -------------------------------------------------------------------------------
def load_dicom_image():
    global filename, img, img_original, mostrar
    filename = filedialog.askopenfilename()
    read_dicom_image()

def read_dicom_image():
    global img, img_original, img_mejor, mostrar, filename
    delete_histogram()
    if filename:
        dicom_data = pydicom.dcmread(filename)
        img_original = cv2.normalize(dicom_data.pixel_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        img_high_res = dicom_data.pixel_array.astype(np.float32)
        img_mejor = img_high_res.copy()
        img = img_original.copy()

        root.update()
        ancho_canva1 = canva_img_original.winfo_width()
        alto_canva1 = canva_img_original.winfo_height()
    
        # Show the image
        pilimg= Image.fromarray(img).convert("L")
        pilimg = pilimg.resize((ancho_canva1, alto_canva1), Image.Resampling.LANCZOS)
        mostrar = ImageTk.PhotoImage(pilimg)
        canva_img_original.delete("all")
        canva_img_original.create_image(ancho_canva1 // 2, alto_canva1 // 2, anchor='center', image=mostrar)
        canva_img_original.image = mostrar
        imprimir()

#IMPRESIÓN DE EDICIONES
def imprimir():
    canva_img_editada.update()
    ancho_canva2 = canva_img_editada.winfo_width()
    alto_canva2 = canva_img_editada.winfo_height()
    pilimg= Image.fromarray(img).convert("L")
    pilimg = pilimg.resize((ancho_canva2, alto_canva2), Image.Resampling.LANCZOS)
    mostrar_editada = ImageTk.PhotoImage(pilimg)
    canva_img_editada.delete("all")
    canva_img_editada.create_image(ancho_canva2 // 2, alto_canva2 // 2, anchor='center', image=mostrar_editada)
    canva_img_editada.image = mostrar_editada
    
#HISTOGRAMA-----------------------------------------------------------------------------------------

def histogram():
    global img, img_original, mostrar_histograma
    fig = Figure(figsize=(8,8), dpi=100, facecolor="#1a1a1a")
    fig.set_facecolor("#1a1a1a")
    ejex=np.arange(256)
    if img_original is not None:
        hist_original = cv2.calcHist([img_original], [0], None, [256], [0, 256]).ravel()
        ax1 = fig.add_subplot(2, 1, 1, facecolor="#1a1a1a")
        ax1.set_facecolor("#2a2a2a")
        ax1.tick_params(colors='white', labelsize=8)
        ax1.bar(ejex,hist_original, color='green', linewidth=2)
        ax1.set_xlabel("Intensidad de píxel", color="#43CD31")
        ax1.set_ylabel("Número de píxeles", color="#43CD31")
        ax1.grid(True, linewidth=0.5, color='gray', alpha=0.5)
        ax1.set_title("Histograma de la imagen", color="#43CD31")

        hist2= cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        ax2 = fig.add_subplot(2, 1, 2, facecolor="#1a1a1a")
        ax2.set_facecolor("#2a2a2a")
        ax2.tick_params(colors='white', labelsize=8)
        ax2.bar(ejex,hist2, color='green', linewidth=2)
        ax2.set_xlabel("Intensidad de píxel", color="#43CD31")
        ax2.set_ylabel("Número de píxeles", color="#43CD31")
        ax2.grid(True, linewidth=0.5, color='gray', alpha=0.5)
        ax2.set_title("Histograma de la imagen editada", color="#43CD31")

        fig.tight_layout(pad=3.0)

        canvas_matplot = FigureCanvasTkAgg(fig, master=canva_img_original) 
        canvas_matplot.draw()
    
        hist_widget = canvas_matplot.get_tk_widget()
        hist_widget.pack(fill="both", expand=True)

def delete_histogram():
    for widget in canva_img_original.winfo_children():
        widget.destroy()
        
#PREPROCESAMIENTO------------------------------------------------------------------------------------

def ecualizar():
    global img
    if img is not None:
        # Ecualizar histograma de la imagen actual
        img_ecualizada = cv2.equalizeHist(img)
        img = img_ecualizada.copy()
        imprimir()


#PROCESAMIENTO --------------------------------------------------------------------------
def resta():
    global img
    img_blur = cv2.blur(img,(5,5))
    img = cv2.subtract(img,img_blur).copy()
    imprimir()

def paso_bajo():
    global img
    img_blur = cv2.medianBlur(img,5)
    img = img_blur.copy()
    imprimir()

def gaussiano ():
    global img_mejor, img_gauss, img
    img_gauss= cv2.GaussianBlur(img_mejor,(5,5), 0)
    img_low_res = cv2.normalize(img_gauss, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img=img_low_res.copy()
    imprimir()

#como por lo visto suelen usar Laplace despues de Gauss tanto que le pusieron un nombre, lo incluiré (LOG= LAPLACIAN OF GAUSSIAN)
def LoG ():
    global img, img_mejor
    img_gauss= cv2.GaussianBlur(img_mejor,(5,5), 0)
    img_laplace = cv2.Laplacian(img_gauss, cv2.CV_32F, ksize=3)
    img_low_res = cv2.normalize(np.absolute(img_laplace), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img=img_low_res.copy()
    imprimir()

def laplace():
    global img_mejor, img
    img_laplace = cv2.Laplacian(img_mejor, cv2.CV_32F, ksize=3)
    img_low_res = cv2.normalize(np.absolute(img_laplace), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img=img_low_res.copy()
    imprimir()
    
def sobel ():
    global img_mejor, img
    sobelx = cv2.Sobel(img_mejor, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_mejor, cv2.CV_32F, 0, 1, ksize=3)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    img_sobel = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    img_low_res = cv2.normalize(img_sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = img_low_res.copy()
    imprimir()

def respaldar_imagen_si_es_necesario():
    global img, img_antes_de_segmentar
    if img_antes_de_segmentar is None and img is not None:
        img_antes_de_segmentar = img.copy()

def umbraladaptativo(slider):
    global img, img_antes_de_segmentar, invertir
    respaldar_imagen_si_es_necesario()
    bloque=int(slider)
    if bloque % 2 == 0:
        bloque += 1  
    if bloque < 3:
        bloque = 3
    if invertir == False:
        mean = cv2.adaptiveThreshold(img_antes_de_segmentar, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bloque, 15)
    elif invertir == True:
        mean = cv2.adaptiveThreshold(img_antes_de_segmentar, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bloque, 15)
    img = mean.copy()
    imprimir()

def invertir_threshold():
    global invertir
    invertir = not invertir
    
#LOGICAS AND/OR
def bitwise_AND():
    global img, img_original
    if img is not None:
        img_and= cv2.bitwise_and(img_original, img)
        img = img_and.copy()
        imprimir()

def bitwise_OR():
    global img, img_original
    if img is not None:
        img_or= cv2.bitwise_or(img_original, img)
        img = img_or.copy()
        imprimir()


#FOURIER --------------------------------------------------------------------------------------
#MAGNITUD 
def magnitud():
    global img
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitud = 20*np.log(np.abs(fshift))
    img = magnitud.copy()
    imprimir()

#DOMINIO DE FRECUENCIA
def frecuencia():
    global img
    f = np.fft.fft2(img)
    shift = np.fft.fftshift(f) 
    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2
    radio = 40
    mask = np.zeros((rows,cols),np.uint8)
    y,x = np.ogrid[:rows,:cols]
    distancia = np.sqrt((x-ccol)**2+(y-crow)**2)
    mask[distancia<=radio]=1
    filtrado = shift*mask
    ishift = np.fft.ifftshift(filtrado)
    sh = np.fft.ifft2(ishift)
    sh = np.abs(sh)
    img = sh.copy()
    imprimir()




#BOTONES---------------------------------------------------------------------------------------
#IMAGEN DICOM 
load_image_btn = tk.Button(button_frame_top, text="Buscar en mi dispositivo", command=load_dicom_image)
load_image_btn.pack(side=tk.LEFT, padx=5, pady=5)

#REGRESAR A LA IMAGEN ORIGINAL
reset_image_btn = tk.Button(button_frame_top, text="Reset", command =read_dicom_image)
reset_image_btn.pack(side=tk.LEFT, padx=5, pady=5)

#INSERTAR HISTOGRAMA
histogram_btn = tk.Button(button_frame_top, text="Mostrar histograma", command=histogram)
histogram_btn.pack(side=tk.LEFT, padx=5, pady=5)

#TOGGLE (EN REALIDAD BORRAR PERO LOL) HISTOGRAMA
toggle_histograma=tk.Button(button_frame_top, text="Eliminar histograma", command=delete_histogram)
toggle_histograma.pack(side=tk.LEFT, padx=5, pady=5)

#INVERTIR
invertir_check = tk.Button(button_frame_top, text="INVERTIR", command=invertir_threshold)
invertir_check.pack(side=tk.LEFT, padx=5, pady=5)

#AND
and_btn = tk.Button(button_frame_top, text="AND", command=bitwise_AND)
and_btn.pack(side=tk.LEFT, padx=5, pady=5)

#OR
or_btn = tk.Button(button_frame_top, text="OR", command=bitwise_OR)
or_btn.pack(side=tk.LEFT, padx=5, pady=5)

#ECUALIZAR
ecualizar_btn = tk.Button(button_frame_bottom, text= "Ecualizar", command=ecualizar)
ecualizar_btn.pack(side=tk.LEFT, padx=5, pady=5)

#RESTA
resta_btn = tk.Button(button_frame_bottom, text="Suavizado",command=resta)
resta_btn.pack(side=tk.LEFT, padx=5, pady=5)

#PASO BAJO
paso_bajo_btn = tk.Button(button_frame_bottom, text="Filtro P. Bajo",command=paso_bajo)
paso_bajo_btn.pack(side=tk.LEFT, padx=5, pady=5)

#GUASSIANO
gaussiano_btn = tk.Button(button_frame_bottom, text="Paso Bajo Gauss" ,command=gaussiano)
gaussiano_btn.pack(side=tk.LEFT, padx=5, pady=5)

#LAPLACE
laplace_btn = tk.Button(button_frame_bottom, text="Paso A. Laplace",command=laplace)
laplace_btn.pack(side=tk.LEFT, padx=5, pady=5)

#SOBEL
sobel_btn = tk.Button(button_frame_bottom, text="Paso A. Sobel",command=sobel)
sobel_btn.pack(side=tk.LEFT, padx=5, pady=5)

#LOG
LoG_btn = tk.Button(button_frame_bottom, text="Laplaciano de Gauss",command=LoG)
LoG_btn.pack(side=tk.LEFT, padx=5, pady=5)

#UMBRALIZACIÓN
escala = tk.Scale(button_frame_bottom, from_= 0, to = 256, orient="horizontal", length = 150, width=10, command=umbraladaptativo)
escala.pack(side=tk.LEFT, padx=5, pady=5)

#FRECUENCIA
domboton = tk.Button(button_frame_bottom, text="Paso B. Fourier", command=frecuencia)
domboton.pack(side=tk.LEFT, padx=5, pady=5)

#MAGNITUD
magnitu = tk.Button(button_frame_bottom, text="Espectro de Frec", command=magnitud)
magnitu.pack(side=tk.LEFT, padx=5, pady=5)


#rickrolled

def rickrolled():
    # URL del video de YouTube que quieras mostrar (por ejemplo, un tutorial de OpenCV o DICOM)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    open(url)
rick_btn = tk.Button( button_frame_top, text="Tutorial", command= rickrolled)
rick_btn.pack(side=tk.LEFT, padx=5, pady=5)


root.mainloop()


