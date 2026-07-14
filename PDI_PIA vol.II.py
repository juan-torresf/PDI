import os
import webbrowser
import tkinter as tk

from tkinter import filedialog, messagebox, ttk

import cv2
import kagglehub
import numpy as np
import pydicom

from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# INTEGRANTES DEL EQUIPO 4
# =============================================================================
# Benito Woo Zozaya - 2177866
# Karen Melissa Morales Moya - 2043909
# Carlos Arturo Arguelles Segura - 2096281
# Juan Gerardo Torres Flores - 2074382


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================
COLOR_FONDO = '#082338'
COLOR_PANEL = '#102f46'
COLOR_PANEL_SECUNDARIO = '#16405c'
COLOR_ACENTO = '#43cd31'
COLOR_TEXTO = '#ffffff'
COLOR_GRAFICA = '#37d67a'
COLOR_CANVAS = '#050b10'

TAMANO_FILTRO = 5
RADIO_FOURIER = 40
RETRASO_HISTOGRAMA_MS = 100


# =============================================================================
# VARIABLES DE ESTADO
# =============================================================================
filename = None

img = None
img_original = None
img_mejor = None
img_gauss = None
img_antes_de_segmentar = None

mostrar_original = None
mostrar_editada = None

invertir = False

histograma_visible = False
histograma_canvas = None
histograma_figura = None
histograma_actualizacion_pendiente = None

path_dataset = None


# =============================================================================
# FUNCIONES GENERALES
# =============================================================================
def actualizar_estado(texto):
    variable_estado.set(texto)
    root.update_idletasks()


def validar_imagen():
    if img is None:
        messagebox.showwarning(
            'Imagen no disponible',
            'Primero debes cargar una imagen DICOM.'
        )
        return False

    return True


def convertir_a_uint8(imagen):
    """
    Convierte cualquier arreglo de imagen a uint8 para mostrarlo correctamente.

    Esto evita problemas cuando una operación produce valores float,
    negativos o fuera del intervalo de 0 a 255.
    """
    if imagen is None:
        return None

    imagen_np = np.asarray(imagen)

    if imagen_np.dtype == np.uint8:
        return imagen_np.copy()

    if not np.isfinite(imagen_np).all():
        imagen_np = np.nan_to_num(
            imagen_np,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

    minimo = float(np.min(imagen_np))
    maximo = float(np.max(imagen_np))

    if minimo == maximo:
        return np.zeros(imagen_np.shape, dtype=np.uint8)

    return cv2.normalize(
        imagen_np,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )


def preparar_imagen_grises(imagen):
    """
    Garantiza que la imagen se encuentre en escala de grises y en uint8.
    """
    imagen_uint8 = convertir_a_uint8(imagen)

    if imagen_uint8 is None:
        return None

    if imagen_uint8.ndim == 3:
        imagen_uint8 = cv2.cvtColor(
            imagen_uint8,
            cv2.COLOR_BGR2GRAY
        )

    return imagen_uint8


def obtener_tamano_canvas(canvas):
    """
    Obtiene un tamaño válido para el canvas.
    """
    canvas.update_idletasks()

    ancho = max(canvas.winfo_width(), 100)
    alto = max(canvas.winfo_height(), 100)

    return ancho, alto


def ajustar_imagen_al_canvas(imagen, canvas):
    """
    Ajusta la imagen al canvas conservando su proporción.
    """
    imagen_uint8 = preparar_imagen_grises(imagen)

    if imagen_uint8 is None:
        return None

    ancho_canvas, alto_canvas = obtener_tamano_canvas(canvas)

    pil_image = Image.fromarray(imagen_uint8).convert('L')

    ancho_imagen, alto_imagen = pil_image.size

    escala = min(
        ancho_canvas / ancho_imagen,
        alto_canvas / alto_imagen
    )

    nuevo_ancho = max(int(ancho_imagen * escala), 1)
    nuevo_alto = max(int(alto_imagen * escala), 1)

    pil_image = pil_image.resize(
        (nuevo_ancho, nuevo_alto),
        Image.Resampling.LANCZOS
    )

    return pil_image


def mostrar_imagen_en_canvas(canvas, imagen, etiqueta):
    """
    Muestra una imagen centrada en un canvas.
    """
    pil_image = ajustar_imagen_al_canvas(imagen, canvas)

    if pil_image is None:
        return

    photo_image = ImageTk.PhotoImage(pil_image)

    canvas.delete('imagen')
    canvas.create_image(
        canvas.winfo_width() // 2,
        canvas.winfo_height() // 2,
        anchor='center',
        image=photo_image,
        tags='imagen'
    )

    canvas.image = photo_image
    etiqueta.config(
        text=f'{pil_image.width} x {pil_image.height} píxeles'
    )


def reiniciar_respaldo_segmentacion():
    global img_antes_de_segmentar

    img_antes_de_segmentar = None


# =============================================================================
# DESCARGA DE RECURSOS
# =============================================================================
def descargar_recursos_kaggle():
    global path_dataset

    try:
        path_dataset = kagglehub.dataset_download(
            'benitowoo/pancreas-and-project-images'
        )

        print('Ruta de los archivos del dataset:', path_dataset)

    except Exception as error:
        path_dataset = None

        print(
            'No fue posible descargar los recursos de Kaggle:',
            error
        )


# =============================================================================
# CARGA DE IMAGEN DICOM
# =============================================================================
def seleccionar_imagen_dicom():
    global filename

    archivo = filedialog.askopenfilename(
        title='Seleccionar imagen DICOM',
        filetypes=[
            ('Archivos DICOM', '*.dcm'),
            ('Todos los archivos', '*.*')
        ]
    )

    if not archivo:
        return

    filename = archivo
    leer_imagen_dicom()


def leer_imagen_dicom():
    global img
    global img_original
    global img_mejor
    global img_gauss
    global img_antes_de_segmentar
    global invertir

    if not filename:
        messagebox.showwarning(
            'Archivo no seleccionado',
            'Primero selecciona una imagen DICOM.'
        )
        return

    try:
        actualizar_estado('Cargando imagen DICOM...')

        dicom_data = pydicom.dcmread(filename)
        pixel_array = dicom_data.pixel_array.astype(np.float32)

        img_original = cv2.normalize(
            pixel_array,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U
        )

        img_mejor = pixel_array.copy()
        img = img_original.copy()
        img_gauss = None
        img_antes_de_segmentar = None
        invertir = False

        texto_invertir.set('Invertir umbral: no')

        mostrar_imagen_original()
        imprimir()

        actualizar_estado(
            f'Imagen cargada: {os.path.basename(filename)}'
        )

    except Exception as error:
        actualizar_estado('Error al cargar la imagen')

        messagebox.showerror(
            'Error de lectura',
            f'No fue posible leer la imagen DICOM.\n\nDetalle: {error}'
        )


def mostrar_imagen_original():
    if img_original is None:
        return

    mostrar_imagen_en_canvas(
        canva_img_original,
        img_original,
        etiqueta_original_info
    )


def imprimir():
    """
    Actualiza la imagen editada y solicita la actualización del histograma.

    Esta función es llamada después de cada filtro. Por esta razón, si el
    histograma está visible, se actualiza automáticamente.
    """
    if img is None:
        return

    mostrar_imagen_en_canvas(
        canva_img_editada,
        img,
        etiqueta_editada_info
    )

    programar_actualizacion_histograma()


def restablecer_imagen():
    global img
    global img_mejor
    global img_gauss
    global img_antes_de_segmentar
    global invertir

    if img_original is None:
        messagebox.showwarning(
            'Imagen no disponible',
            'Primero debes cargar una imagen DICOM.'
        )
        return

    img = img_original.copy()
    img_mejor = img_original.astype(np.float32)
    img_gauss = None
    img_antes_de_segmentar = None
    invertir = False

    texto_invertir.set('Invertir umbral: no')

    imprimir()
    actualizar_estado('Imagen restablecida')


# =============================================================================
# HISTOGRAMA
# =============================================================================
def configurar_ejes_histograma(ejes, titulo):
    ejes.set_facecolor('#17212b')

    ejes.tick_params(
        colors=COLOR_TEXTO,
        labelsize=8
    )

    ejes.set_xlabel(
        'Intensidad de píxel',
        color=COLOR_ACENTO
    )

    ejes.set_ylabel(
        'Número de píxeles',
        color=COLOR_ACENTO
    )

    ejes.set_title(
        titulo,
        color=COLOR_ACENTO,
        fontsize=11,
        fontweight='bold'
    )

    ejes.grid(
        True,
        linewidth=0.5,
        color='#808080',
        alpha=0.35
    )

    for borde in ejes.spines.values():
        borde.set_color('#5e7182')


def mostrar_histograma():
    global histograma_visible

    if not validar_imagen():
        return

    histograma_visible = True
    actualizar_histograma()
    actualizar_estado('Histograma visible')


def actualizar_histograma():
    global histograma_canvas
    global histograma_figura
    global histograma_actualizacion_pendiente

    histograma_actualizacion_pendiente = None

    if not histograma_visible:
        return

    if img_original is None or img is None:
        return

    imagen_original_hist = preparar_imagen_grises(img_original)
    imagen_editada_hist = preparar_imagen_grises(img)

    hist_original = cv2.calcHist(
        [imagen_original_hist],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    hist_editada = cv2.calcHist(
        [imagen_editada_hist],
        [0],
        None,
        [256],
        [0, 256]
    ).ravel()

    valores_x = np.arange(256)

    if histograma_figura is None:
        histograma_figura = Figure(
            figsize=(7, 6),
            dpi=100,
            facecolor=COLOR_CANVAS
        )

    histograma_figura.clear()

    eje_original = histograma_figura.add_subplot(
        2,
        1,
        1,
        facecolor='#17212b'
    )

    eje_editada = histograma_figura.add_subplot(
        2,
        1,
        2,
        facecolor='#17212b'
    )

    configurar_ejes_histograma(
        eje_original,
        'Histograma de la imagen original'
    )

    configurar_ejes_histograma(
        eje_editada,
        'Histograma de la imagen editada'
    )

    eje_original.plot(
        valores_x,
        hist_original,
        color=COLOR_GRAFICA,
        linewidth=1.2
    )

    eje_original.fill_between(
        valores_x,
        hist_original,
        color=COLOR_GRAFICA,
        alpha=0.25
    )

    eje_editada.plot(
        valores_x,
        hist_editada,
        color='#48a9ff',
        linewidth=1.2
    )

    eje_editada.fill_between(
        valores_x,
        hist_editada,
        color='#48a9ff',
        alpha=0.25
    )

    eje_original.set_xlim(0, 255)
    eje_editada.set_xlim(0, 255)

    histograma_figura.tight_layout(pad=2.0)

    if histograma_canvas is None:
        histograma_canvas = FigureCanvasTkAgg(
            histograma_figura,
            master=contenedor_histograma
        )

        widget_histograma = histograma_canvas.get_tk_widget()
        widget_histograma.pack(
            fill='both',
            expand=True
        )

    histograma_canvas.draw_idle()

    contenedor_imagen_original.grid_remove()
    contenedor_histograma.grid()


def programar_actualizacion_histograma():
    """
    Actualiza el histograma solamente si está visible.

    Se usa un pequeño retraso para no redibujar decenas de veces mientras
    el usuario mueve rápidamente el control de umbral.
    """
    global histograma_actualizacion_pendiente

    if not histograma_visible:
        return

    if histograma_actualizacion_pendiente is not None:
        root.after_cancel(histograma_actualizacion_pendiente)

    histograma_actualizacion_pendiente = root.after(
        RETRASO_HISTOGRAMA_MS,
        actualizar_histograma
    )


def eliminar_histograma():
    global histograma_visible
    global histograma_canvas
    global histograma_figura
    global histograma_actualizacion_pendiente

    histograma_visible = False

    if histograma_actualizacion_pendiente is not None:
        root.after_cancel(histograma_actualizacion_pendiente)
        histograma_actualizacion_pendiente = None

    if histograma_canvas is not None:
        histograma_canvas.get_tk_widget().destroy()
        histograma_canvas = None

    histograma_figura = None

    contenedor_histograma.grid_remove()
    contenedor_imagen_original.grid()

    mostrar_imagen_original()
    actualizar_estado('Histograma oculto')


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================
def ecualizar():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_uint8 = preparar_imagen_grises(img)
    img = cv2.equalizeHist(imagen_uint8)

    imprimir()
    actualizar_estado('Ecualización de histograma aplicada')


# =============================================================================
# PROCESAMIENTO ESPACIAL
# =============================================================================
def resta():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_uint8 = preparar_imagen_grises(img)
    imagen_suavizada = cv2.blur(
        imagen_uint8,
        (TAMANO_FILTRO, TAMANO_FILTRO)
    )

    img = cv2.subtract(
        imagen_uint8,
        imagen_suavizada
    )

    imprimir()
    actualizar_estado('Resta con imagen suavizada aplicada')


def paso_bajo():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_uint8 = preparar_imagen_grises(img)
    img = cv2.medianBlur(
        imagen_uint8,
        TAMANO_FILTRO
    )

    imprimir()
    actualizar_estado('Filtro de mediana aplicado')


def gaussiano():
    global img
    global img_gauss

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img)

    img_gauss = cv2.GaussianBlur(
        imagen_actual,
        (TAMANO_FILTRO, TAMANO_FILTRO),
        0
    )

    img = img_gauss.copy()

    imprimir()
    actualizar_estado('Filtro gaussiano aplicado')


def laplaciano_de_gauss():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img).astype(np.float32)

    imagen_gauss = cv2.GaussianBlur(
        imagen_actual,
        (TAMANO_FILTRO, TAMANO_FILTRO),
        0
    )

    imagen_laplace = cv2.Laplacian(
        imagen_gauss,
        cv2.CV_32F,
        ksize=3
    )

    img = cv2.normalize(
        np.abs(imagen_laplace),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    imprimir()
    actualizar_estado('Laplaciano de Gauss aplicado')


def laplace():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img).astype(np.float32)

    imagen_laplace = cv2.Laplacian(
        imagen_actual,
        cv2.CV_32F,
        ksize=3
    )

    img = cv2.normalize(
        np.abs(imagen_laplace),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    imprimir()
    actualizar_estado('Filtro Laplace aplicado')


def sobel():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img).astype(np.float32)

    sobel_x = cv2.Sobel(
        imagen_actual,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        imagen_actual,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitud_sobel = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    img = cv2.normalize(
        magnitud_sobel,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    imprimir()
    actualizar_estado('Filtro Sobel aplicado')


# =============================================================================
# UMBRALIZACIÓN
# =============================================================================
def respaldar_imagen_si_es_necesario():
    global img_antes_de_segmentar

    if img_antes_de_segmentar is None and img is not None:
        img_antes_de_segmentar = preparar_imagen_grises(img)


def umbral_adaptativo(valor_slider=None):
    global img

    if img is None:
        return

    respaldar_imagen_si_es_necesario()

    bloque = int(float(escala_umbral.get()))

    if bloque % 2 == 0:
        bloque += 1

    bloque = max(bloque, 3)

    tipo_umbral = (
        cv2.THRESH_BINARY_INV
        if invertir
        else cv2.THRESH_BINARY
    )

    img = cv2.adaptiveThreshold(
        img_antes_de_segmentar,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        tipo_umbral,
        bloque,
        15
    )

    imprimir()

    estado_inversion = 'invertido' if invertir else 'normal'

    actualizar_estado(
        f'Umbral adaptativo: bloque {bloque}, modo {estado_inversion}'
    )


def invertir_threshold():
    global invertir

    if not validar_imagen():
        return

    invertir = not invertir

    texto_invertir.set(
        'Invertir umbral: sí'
        if invertir
        else 'Invertir umbral: no'
    )

    umbral_adaptativo()


# =============================================================================
# OPERACIONES LÓGICAS
# =============================================================================
def bitwise_and():
    global img

    if not validar_imagen():
        return

    imagen_actual = preparar_imagen_grises(img)
    original = preparar_imagen_grises(img_original)

    img = cv2.bitwise_and(
        original,
        imagen_actual
    )

    reiniciar_respaldo_segmentacion()
    imprimir()
    actualizar_estado('Operación AND aplicada')


def bitwise_or():
    global img

    if not validar_imagen():
        return

    imagen_actual = preparar_imagen_grises(img)
    original = preparar_imagen_grises(img_original)

    img = cv2.bitwise_or(
        original,
        imagen_actual
    )

    reiniciar_respaldo_segmentacion()
    imprimir()
    actualizar_estado('Operación OR aplicada')


# =============================================================================
# TRANSFORMADA DE FOURIER
# =============================================================================
def espectro_magnitud():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img).astype(np.float32)

    transformada = np.fft.fft2(imagen_actual)
    transformada_centrada = np.fft.fftshift(transformada)

    espectro = 20 * np.log1p(
        np.abs(transformada_centrada)
    )

    img = cv2.normalize(
        espectro,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    imprimir()
    actualizar_estado('Espectro de magnitud calculado')


def filtro_frecuencia():
    global img

    if not validar_imagen():
        return

    reiniciar_respaldo_segmentacion()

    imagen_actual = preparar_imagen_grises(img).astype(np.float32)

    transformada = np.fft.fft2(imagen_actual)
    transformada_centrada = np.fft.fftshift(transformada)

    filas, columnas = imagen_actual.shape
    centro_fila = filas // 2
    centro_columna = columnas // 2

    coordenada_y, coordenada_x = np.ogrid[
        :filas,
        :columnas
    ]

    distancia = np.sqrt(
        (coordenada_x - centro_columna) ** 2
        + (coordenada_y - centro_fila) ** 2
    )

    mascara = np.zeros(
        (filas, columnas),
        dtype=np.float32
    )

    mascara[distancia <= RADIO_FOURIER] = 1.0

    transformada_filtrada = transformada_centrada * mascara

    transformada_inversa_centrada = np.fft.ifftshift(
        transformada_filtrada
    )

    imagen_filtrada = np.fft.ifft2(
        transformada_inversa_centrada
    )

    imagen_filtrada = np.abs(imagen_filtrada)

    img = cv2.normalize(
        imagen_filtrada,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U
    )

    imprimir()
    actualizar_estado('Filtro paso bajo de Fourier aplicado')


# =============================================================================
# OTROS
# =============================================================================
def abrir_tutorial():
    url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    webbrowser.open(url)


def actualizar_imagenes_al_redimensionar(event=None):
    """
    Vuelve a ajustar las imágenes cuando cambia el tamaño de la ventana.
    """
    if img_original is not None and not histograma_visible:
        mostrar_imagen_original()

    if img is not None:
        mostrar_imagen_en_canvas(
            canva_img_editada,
            img,
            etiqueta_editada_info
        )


# =============================================================================
# INTERFAZ GRÁFICA
# =============================================================================
root = tk.Tk()

root.title(
    'Software de Procesamiento de Imágenes Médicas - Equipo 4'
)

root.configure(bg=COLOR_FONDO)
root.minsize(1100, 700)

try:
    root.state('zoomed')
except tk.TclError:
    root.geometry('1400x850')

root.bind(
    '<Escape>',
    lambda event: root.destroy()
)


# =============================================================================
# ESTILOS
# =============================================================================
style = ttk.Style()

try:
    style.theme_use('clam')
except tk.TclError:
    pass

style.configure(
    'Principal.TFrame',
    background=COLOR_FONDO
)

style.configure(
    'Panel.TFrame',
    background=COLOR_PANEL
)

style.configure(
    'Titulo.TLabel',
    background=COLOR_FONDO,
    foreground=COLOR_TEXTO,
    font=('Segoe UI', 22, 'bold')
)

style.configure(
    'Subtitulo.TLabel',
    background=COLOR_PANEL,
    foreground=COLOR_ACENTO,
    font=('Segoe UI', 12, 'bold')
)

style.configure(
    'Info.TLabel',
    background=COLOR_PANEL,
    foreground='#b9c7d4',
    font=('Segoe UI', 9)
)

style.configure(
    'Estado.TLabel',
    background='#061824',
    foreground=COLOR_TEXTO,
    font=('Segoe UI', 9)
)

style.configure(
    'Accion.TButton',
    font=('Segoe UI', 9, 'bold'),
    padding=(10, 7),
    background=COLOR_ACENTO,
    foreground='#07131c'
)

style.map(
    'Accion.TButton',
    background=[
        ('active', '#63e653'),
        ('pressed', '#2fa823')
    ]
)

style.configure(
    'Secundario.TButton',
    font=('Segoe UI', 9),
    padding=(10, 7),
    background=COLOR_PANEL_SECUNDARIO,
    foreground=COLOR_TEXTO
)

style.map(
    'Secundario.TButton',
    background=[
        ('active', '#205678'),
        ('pressed', '#0e2c40')
    ]
)

style.configure(
    'TNotebook',
    background=COLOR_PANEL,
    borderwidth=0
)

style.configure(
    'TNotebook.Tab',
    font=('Segoe UI', 9, 'bold'),
    padding=(12, 8),
    background=COLOR_PANEL_SECUNDARIO,
    foreground=COLOR_TEXTO
)

style.map(
    'TNotebook.Tab',
    background=[
        ('selected', COLOR_ACENTO)
    ],
    foreground=[
        ('selected', '#07131c')
    ]
)


# =============================================================================
# CONTENEDOR PRINCIPAL
# =============================================================================
contenedor_principal = ttk.Frame(
    root,
    style='Principal.TFrame',
    padding=12
)

contenedor_principal.pack(
    fill='both',
    expand=True
)

contenedor_principal.columnconfigure(
    0,
    weight=1
)

contenedor_principal.rowconfigure(
    1,
    weight=1
)


# =============================================================================
# ENCABEZADO
# =============================================================================
encabezado = ttk.Frame(
    contenedor_principal,
    style='Principal.TFrame'
)

encabezado.grid(
    row=0,
    column=0,
    sticky='ew',
    pady=(0, 10)
)

encabezado.columnconfigure(
    1,
    weight=1
)

contenedor_logo = ttk.Frame(
    encabezado,
    style='Principal.TFrame'
)

contenedor_logo.grid(
    row=0,
    column=0,
    sticky='w'
)

if path_dataset:
    ruta_logo = os.path.join(
        path_dataset,
        'AMAZE.png'
    )

    if os.path.exists(ruta_logo):
        try:
            imagen_logo_pil = Image.open(ruta_logo)
            imagen_logo_pil = imagen_logo_pil.resize(
                (64, 64),
                Image.Resampling.LANCZOS
            )

            imagen_logo = ImageTk.PhotoImage(
                imagen_logo_pil
            )

            etiqueta_logo = tk.Label(
                contenedor_logo,
                image=imagen_logo,
                bg=COLOR_FONDO
            )

            etiqueta_logo.image = imagen_logo

            etiqueta_logo.pack(
                padx=(0, 12)
            )

        except Exception as error:
            print('No fue posible cargar el logotipo:', error)

titulo = ttk.Label(
    encabezado,
    text='Procesamiento de Imágenes Médicas',
    style='Titulo.TLabel'
)

titulo.grid(
    row=0,
    column=1,
    sticky='w'
)

subtitulo = tk.Label(
    encabezado,
    text='PIA · Equipo 4',
    bg=COLOR_FONDO,
    fg=COLOR_ACENTO,
    font=('Segoe UI', 11)
)

subtitulo.grid(
    row=1,
    column=1,
    sticky='w'
)


# =============================================================================
# ÁREA DE IMÁGENES
# =============================================================================
area_imagenes = ttk.Frame(
    contenedor_principal,
    style='Principal.TFrame'
)

area_imagenes.grid(
    row=1,
    column=0,
    sticky='nsew'
)

area_imagenes.columnconfigure(
    0,
    weight=1,
    uniform='imagenes'
)

area_imagenes.columnconfigure(
    1,
    weight=1,
    uniform='imagenes'
)

area_imagenes.rowconfigure(
    0,
    weight=1
)


# Panel de imagen editada
panel_editada = ttk.Frame(
    area_imagenes,
    style='Panel.TFrame',
    padding=8
)

panel_editada.grid(
    row=0,
    column=0,
    sticky='nsew',
    padx=(0, 6)
)

panel_editada.columnconfigure(
    0,
    weight=1
)

panel_editada.rowconfigure(
    1,
    weight=1
)

titulo_editada = ttk.Label(
    panel_editada,
    text='Imagen procesada',
    style='Subtitulo.TLabel'
)

titulo_editada.grid(
    row=0,
    column=0,
    sticky='w',
    pady=(0, 6)
)

canva_img_editada = tk.Canvas(
    panel_editada,
    bg=COLOR_CANVAS,
    highlightthickness=1,
    highlightbackground=COLOR_ACENTO
)

canva_img_editada.grid(
    row=1,
    column=0,
    sticky='nsew'
)

etiqueta_editada_info = ttk.Label(
    panel_editada,
    text='Sin imagen',
    style='Info.TLabel'
)

etiqueta_editada_info.grid(
    row=2,
    column=0,
    sticky='w',
    pady=(5, 0)
)


# Panel de imagen original e histograma
panel_original = ttk.Frame(
    area_imagenes,
    style='Panel.TFrame',
    padding=8
)

panel_original.grid(
    row=0,
    column=1,
    sticky='nsew',
    padx=(6, 0)
)

panel_original.columnconfigure(
    0,
    weight=1
)

panel_original.rowconfigure(
    1,
    weight=1
)

titulo_original = ttk.Label(
    panel_original,
    text='Imagen original / Histograma',
    style='Subtitulo.TLabel'
)

titulo_original.grid(
    row=0,
    column=0,
    sticky='w',
    pady=(0, 6)
)

contenedor_visual_original = ttk.Frame(
    panel_original,
    style='Panel.TFrame'
)

contenedor_visual_original.grid(
    row=1,
    column=0,
    sticky='nsew'
)

contenedor_visual_original.columnconfigure(
    0,
    weight=1
)

contenedor_visual_original.rowconfigure(
    0,
    weight=1
)

contenedor_imagen_original = ttk.Frame(
    contenedor_visual_original,
    style='Panel.TFrame'
)

contenedor_imagen_original.grid(
    row=0,
    column=0,
    sticky='nsew'
)

contenedor_imagen_original.columnconfigure(
    0,
    weight=1
)

contenedor_imagen_original.rowconfigure(
    0,
    weight=1
)

canva_img_original = tk.Canvas(
    contenedor_imagen_original,
    bg=COLOR_CANVAS,
    highlightthickness=1,
    highlightbackground=COLOR_ACENTO
)

canva_img_original.grid(
    row=0,
    column=0,
    sticky='nsew'
)

contenedor_histograma = ttk.Frame(
    contenedor_visual_original,
    style='Panel.TFrame'
)

contenedor_histograma.grid(
    row=0,
    column=0,
    sticky='nsew'
)

contenedor_histograma.grid_remove()

etiqueta_original_info = ttk.Label(
    panel_original,
    text='Sin imagen',
    style='Info.TLabel'
)

etiqueta_original_info.grid(
    row=2,
    column=0,
    sticky='w',
    pady=(5, 0)
)


# =============================================================================
# PANEL DE CONTROLES
# =============================================================================
panel_controles = ttk.Frame(
    contenedor_principal,
    style='Panel.TFrame',
    padding=8
)

panel_controles.grid(
    row=2,
    column=0,
    sticky='ew',
    pady=(10, 0)
)

panel_controles.columnconfigure(
    0,
    weight=1
)

notebook = ttk.Notebook(
    panel_controles
)

notebook.grid(
    row=0,
    column=0,
    sticky='ew'
)

pestana_archivo = ttk.Frame(
    notebook,
    style='Panel.TFrame',
    padding=8
)

pestana_filtros = ttk.Frame(
    notebook,
    style='Panel.TFrame',
    padding=8
)

pestana_segmentacion = ttk.Frame(
    notebook,
    style='Panel.TFrame',
    padding=8
)

pestana_fourier = ttk.Frame(
    notebook,
    style='Panel.TFrame',
    padding=8
)

notebook.add(
    pestana_archivo,
    text='Archivo e histograma'
)

notebook.add(
    pestana_filtros,
    text='Filtros espaciales'
)

notebook.add(
    pestana_segmentacion,
    text='Segmentación'
)

notebook.add(
    pestana_fourier,
    text='Fourier'
)


# =============================================================================
# BOTONES DE ARCHIVO E HISTOGRAMA
# =============================================================================
boton_cargar = ttk.Button(
    pestana_archivo,
    text='Buscar imagen DICOM',
    command=seleccionar_imagen_dicom,
    style='Accion.TButton'
)

boton_cargar.pack(
    side='left',
    padx=4
)

boton_reset = ttk.Button(
    pestana_archivo,
    text='Restablecer',
    command=restablecer_imagen,
    style='Secundario.TButton'
)

boton_reset.pack(
    side='left',
    padx=4
)

boton_histograma = ttk.Button(
    pestana_archivo,
    text='Mostrar histograma',
    command=mostrar_histograma,
    style='Accion.TButton'
)

boton_histograma.pack(
    side='left',
    padx=4
)

boton_eliminar_histograma = ttk.Button(
    pestana_archivo,
    text='Ocultar histograma',
    command=eliminar_histograma,
    style='Secundario.TButton'
)

boton_eliminar_histograma.pack(
    side='left',
    padx=4
)

boton_tutorial = ttk.Button(
    pestana_archivo,
    text='Tutorial',
    command=abrir_tutorial,
    style='Secundario.TButton'
)

boton_tutorial.pack(
    side='left',
    padx=4
)


# =============================================================================
# BOTONES DE FILTROS
# =============================================================================
controles_filtros = [
    ('Ecualizar', ecualizar),
    ('Resta suavizada', resta),
    ('Filtro de mediana', paso_bajo),
    ('Gaussiano', gaussiano),
    ('Laplace', laplace),
    ('Sobel', sobel),
    ('Laplaciano de Gauss', laplaciano_de_gauss)
]

for texto_boton, comando_boton in controles_filtros:
    boton = ttk.Button(
        pestana_filtros,
        text=texto_boton,
        command=comando_boton,
        style='Secundario.TButton'
    )

    boton.pack(
        side='left',
        padx=4
    )


# =============================================================================
# CONTROLES DE SEGMENTACIÓN
# =============================================================================
texto_invertir = tk.StringVar(
    value='Invertir umbral: no'
)

etiqueta_umbral = ttk.Label(
    pestana_segmentacion,
    text='Tamaño del bloque:',
    style='Subtitulo.TLabel'
)

etiqueta_umbral.pack(
    side='left',
    padx=(4, 8)
)

escala_umbral = tk.Scale(
    pestana_segmentacion,
    from_=3,
    to=255,
    orient='horizontal',
    length=220,
    resolution=2,
    command=umbral_adaptativo,
    bg=COLOR_PANEL,
    fg=COLOR_TEXTO,
    troughcolor=COLOR_PANEL_SECUNDARIO,
    activebackground=COLOR_ACENTO,
    highlightthickness=0
)

escala_umbral.set(15)

escala_umbral.pack(
    side='left',
    padx=4
)

boton_invertir = ttk.Button(
    pestana_segmentacion,
    textvariable=texto_invertir,
    command=invertir_threshold,
    style='Secundario.TButton'
)

boton_invertir.pack(
    side='left',
    padx=4
)

boton_and = ttk.Button(
    pestana_segmentacion,
    text='Operación AND',
    command=bitwise_and,
    style='Secundario.TButton'
)

boton_and.pack(
    side='left',
    padx=4
)

boton_or = ttk.Button(
    pestana_segmentacion,
    text='Operación OR',
    command=bitwise_or,
    style='Secundario.TButton'
)

boton_or.pack(
    side='left',
    padx=4
)


# =============================================================================
# CONTROLES DE FOURIER
# =============================================================================
boton_frecuencia = ttk.Button(
    pestana_fourier,
    text='Filtro paso bajo',
    command=filtro_frecuencia,
    style='Secundario.TButton'
)

boton_frecuencia.pack(
    side='left',
    padx=4
)

boton_magnitud = ttk.Button(
    pestana_fourier,
    text='Espectro de magnitud',
    command=espectro_magnitud,
    style='Secundario.TButton'
)

boton_magnitud.pack(
    side='left',
    padx=4
)


# =============================================================================
# BARRA DE ESTADO
# =============================================================================
variable_estado = tk.StringVar(
    value='Listo. Selecciona una imagen DICOM para comenzar.'
)

barra_estado = ttk.Label(
    contenedor_principal,
    textvariable=variable_estado,
    style='Estado.TLabel',
    anchor='w',
    padding=(10, 5)
)

barra_estado.grid(
    row=3,
    column=0,
    sticky='ew',
    pady=(8, 0)
)


# =============================================================================
# EVENTOS
# =============================================================================
temporizador_redimension = None


def manejar_redimension(event):
    global temporizador_redimension

    if event.widget is not root:
        return

    if temporizador_redimension is not None:
        root.after_cancel(temporizador_redimension)

    temporizador_redimension = root.after(
        150,
        actualizar_imagenes_al_redimensionar
    )


root.bind(
    '<Configure>',
    manejar_redimension
)


# =============================================================================
# INICIO DEL PROGRAMA
# =============================================================================
descargar_recursos_kaggle()

root.mainloop()