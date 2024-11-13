import tkinter as tk
from tkinter import filedialog, messagebox
from ttkthemes import ThemedTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import pandas as pd


def cargar_imagen():
    # Abre la ventana para selelccionar la imagen 
    global tomografia
    file_path = filedialog.askopenfilename(title="Selecciona una imagen", 
                                        filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff")])

    if file_path:
        # Abrir y mostrar la imagen seleccionada
        #img.show()  # Muestra la imagen en la aplicación predeterminada
        imagen = cv2.imread(file_path, 0)
        tomografia = imagen
        return imagen
    else:
        #en caso de que no elija ninguna imagen se muestra un mensaje de advertencia y el programa no prosigue
        print("No se seleccionó ninguna imagen.")
        return None

def plotear_imagen(imagen):
    # Limpia el plot y muestra la imagen
    ax.clear()
    plt.imshow(imagen, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    canvas.draw()

def seleccionar_y_mostrar():
    #selecciono imagen y la muestra
    imagen = cargar_imagen()
    plotear_imagen(imagen)

#funcion para clusterizar la imagen (solo funciona con la imagen)
def kmeans(imagen,k, iteraciones,epsilon):
  copia=imagen.copy()
  pixel_vals = copia.reshape((-1))#se pasa la dimensión de la imagen a no--> NO IMPORTA LA DISTRIBUCIÓN ESPACIAL
  pixel_vals = np.float32(pixel_vals)#El algoritmos nos pide floats de 32 bits
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iteraciones, epsilon) #se para el algoritmo cuando se haya cumplido el epsilon o cuando se hayan acabado las iteraciones,cada attemp itera n veces, epsilon (precision)
  flags = cv2.KMEANS_RANDOM_CENTERS #situa los centroides inicales
  compactness, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, flags)

  center = np.uint8(centers)

  img_1_kmeans = center[labels.flatten()] #Asigna a cada píxel el valor del centro del cluster al que pertenece
  img_1_kmeans = img_1_kmeans.reshape((copia.shape))
  centers=centers.flatten()
  return img_1_kmeans, centers
def binarizar(img_clust, centros):
  umb, img_bin = cv2.threshold(img_clust, max(centros)-10, 255, cv2.THRESH_BINARY)
  return umb, img_bin

# funcion para cargar los labels en un vector
def cargar_labels(path):
   # aca pongan el path a su carpeta del dataset
    train_labels_dir = path
    # me armo una lista
    train_labels_files = [f for f in os.listdir(train_labels_dir) if os.path.isfile(os.path.join(train_labels_dir, f))]
    # creo un dataframe con las rutas de las imagenes
    train_labels_paths = pd.DataFrame({'LabelPath': [os.path.join(train_labels_dir, f) for f in train_labels_files]})
    train_labels = []   
    for path in train_labels_paths['LabelPath']:
    # leo los labels
        with open(path, 'r') as file:
            lines = file.read().splitlines()
            separated_values = [line.split() for line in lines]
            # Convierto los valores a float y aplano la lista
            numeric_values = [float(value) for line in separated_values for value in line]
        train_labels.append(numeric_values)
    return train_labels

#funcion para a partir de la data de labels conseguir area,ratio,dimensiones y coordenadas
def analisis_cajas(labels):
    Areas = []
    Ratios = []
    Coordenadas = []
    Dimensiones = []
    for i in range(len(labels)):
        if len(labels[i])>4:
            for j in range(1,len(labels[i]),5):
                x = labels[i][j]*391
                y = labels[i][j+1]*320
                w = labels[i][j+2]*391
                h = labels[i][j+3]*320
                ratio = w/h
                Ratios.append(ratio)
                Areas.append(w*h)
                Coordenadas.append([x,y])
                Dimensiones.append([w,h])
    return Areas, Ratios, Coordenadas, Dimensiones

#funcion que promedia y busca desvio

def promedios_cajas(Areas, Ratios):
    prom_area = np.mean(Areas)
    prom_ratio = np.mean(Ratios)
    desv_area = np.std(Areas)
    desv_ratio = np.std(Ratios)
    return prom_area,desv_area, prom_ratio, desv_ratio

#funcion de apertura de imagen
def apertura(imgbinaria):
  kernel = np.ones((9, 9), 'uint8')
  erode_img = cv2.erode(imgbinaria, kernel, iterations=1)

  kernel = np.ones((4, 4), 'uint8')
  close_img = cv2.dilate(erode_img, kernel, iterations=1)
  return close_img

#funcion que dibuja las cajas solo si esta dentro de nuestros criterios
def Boxes(img,Prom_A,Prom_R,Desv_R,Cota):
    ax.clear()
    img_cluster, centros=kmeans(img,3,10,0.9)
    imgbin=binarizar(img_cluster,centros)[1]
    img_Canny = cv2.Canny(imgbin, 200, 256)
    contours, _ = cv2.findContours(img_Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copia = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (50+w<x+w<150 or 250+w<x+w<330): #solo dibujo la caja si se encuentra en la posicion de los riñones
            if (Prom_A*(1-Cota) < w*h < Prom_A*(1+Cota)) and (Prom_R - Desv_R < w/h < Prom_R + Desv_R and w< 50 and h<50): #solo dibujo la caja si se encuentra en una dimension y ratio similar al de las cajas dato
                    cv2.rectangle(copia, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(copia, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    canvas.draw()


#funcion para obtener las coordenadas de un click en la pantalla y dibujar un punto en el click
def onclick(event):
    global x, y
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        # Dibuja un punto en las coordenadas
        ax.plot(x, y, 'ro', markersize=2)  
        plt.draw()

#funcion para obtener las coordenadas de un click en la pantalla
def obtener_coordenadas():
    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

#algoritmo de region growing tomando como variables globales las coordenadas del click
def region_growing():
    global tomografia, cid, x, y,region
    if 'x' not in globals() or 'y' not in globals(): # Si no se seleccionó una coordenada previamente se muestra un mensaje de error
        messagebox.showerror("Error", "Por favor, selecciona una coordenada primero.")
        return
    ax.clear()
    fig.canvas.mpl_disconnect(cid)
    seed = (x, y)
    img = tomografia
    mg_cluster, centros=kmeans(img,3,10,0.9)
    img = binarizar(img,centros)[1]
    rows, cols = img.shape
    region = np.zeros_like(img)
    region[seed[1], seed[0]] = 255
    stack = [seed]

    while stack:
        x, y = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and region[ny, nx] == 0:
                if abs(int(img[ny, nx]) - int(img[y, x])) < 100:
                    region[ny, nx] = 255
                    stack.append((nx, ny))
    ax.clear()
    ax.imshow(region, cmap='gray')
    ax.axis('off')
    canvas.draw()
    return region

#funcion para clusterizar las dimensiones de los calculos
#distinta de la funcion kmeans declarada previamente, esta unicamente se utiliza en la data de las dimensiones de los calculos
def kmeans_clusters_data():
    # Stackeamos los datos para obtener datos en 2D
    global widths,heights,centersord,new_labels,centers
    data = np.float32(np.column_stack((widths, heights)))  # Datos en el formato correcto

    # Parámetros de kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5 # número de clusters
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Ordenar los centros en función del eje X
    sorted_indices = np.argsort(centers[:, 0])  # Ordenar por coordenada X
    centersord = centers[sorted_indices]  # Reordenar los centros en base al índice ordenado

    # Crear un nuevo array de etiquetas basado en el orden de los centros
    new_labels = np.zeros_like(labels)

    # Reasignar las etiquetas sin usar enumerate
    for i in range(k):
        old_label = sorted_indices[i]  # Índice original del centroide
        new_labels[labels == old_label] = i  # Reasignar nueva etiqueta (i) a los puntos que corresponden a old_label

#funcion para calcular la distancia entre un punto y un centroide
def distancia(wc,hc,h,w):
  return np.sqrt((hc-h)**2+(wc-w)**2)

#funcion para clasificar el calculo en un grado en funcion a la distancia al centroide
def clasificacion():
    global centersord, grado,centers,region,grado_var
    obtener_coordenadas()
    region = region_growing()
    region_canny = cv2.Canny(region, 10, 256)
    contours, _ = cv2.findContours(region_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
    distancias=np.zeros(len(centers)) 
    for i in range(len(centers)): 
        distancias[i]=distancia(centersord[i][0],centersord[i][1],h,w) 
        min(distancias)
    for i in range(len(distancias)):
        if distancias[i]==min(distancias):
            grado = str(i+1)
            grado_var.set(f"El cálculo es de Grado {i+1}")

#aca escriban el path que les lleva a la carpeta de labels del dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'Dataset/train/labels')  

#variables globales
tomografia = None
grado = "Grado del cálculo: Sin calcular"

#Creacion de la ventana
app = ThemedTk(theme="winxpblue")

#Variables de control
grado_var = tk.StringVar(app,value = grado)
cargar_labels(path)
A,R,C,D = analisis_cajas(cargar_labels(path))
PrA, DesA, PrR, DesR = promedios_cajas(A,R)
palabra = tk.StringVar(app)
entrada = tk.StringVar(app)
Prom_A = tk.DoubleVar( value=PrA)
Prom_R = tk.DoubleVar( value=PrR)
Desv_R = tk.DoubleVar( value=DesR)
Cota = tk.DoubleVar( value=2)
#####

widths = [dim[0] for dim in D]
heights = [dim[1] for dim in D]
#clusterizo las dimensiones de los calculos
kmeans_clusters_data()
#Dimensiones: Ancho x Altura
app.geometry("1000x1000")
#Cambio el color del fondo
app.configure(bg="grey")
#Titulo de la ventana
app.title("Detector de calculos renales")

#prueba de grafico
#creo un frame para los botones
frame_buttons = tk.Frame(app, bg="grey")
frame_buttons.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
#creo un frame para el plot y el label
frame_plot = tk.Frame(app, bg="grey")
frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(6, 5))  # Tamaño del plot
plt.imshow(np.zeros((320,391)), cmap='gray', vmin=0, vmax=255)
ax.axis('off')

# Crear el canvas para mostrar el plot
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack()

# Crear la barra de herramientas 
herramienas = NavigationToolbar2Tk(canvas, frame_plot, pack_toolbar=False)
herramienas.update()
herramienas.pack(anchor='nw', fill='both')

canvas.get_tk_widget().pack()
herramienas.update()
herramienas.pack(anchor='nw', fill='both')

Grade = tk.Label(
    frame_plot,
    textvariable=grado_var,
    text="Grado del calculo renal: ",
    font=("Arial", 16,"bold"),
    bg="blue",
    fg="white",
    relief="raised",
).pack(fill=tk.BOTH, expand=True, anchor='nw')

button_style = {
    "font": ("Arial", 20, "bold"),
    "bg": "#4CAF50",
    "fg": "white",
    "relief": "raised",
    "bd": 5,
    "activebackground": "#45a049",
    "activeforeground": "white",
    "cursor": "hand2"
}

tk.Button(
    frame_buttons,
    text="Cargar imagen",
    command=seleccionar_y_mostrar,
    **button_style
).pack(
    fill=tk.BOTH,
    expand=True,
    pady=5,
    padx=10
)

tk.Button(
    frame_buttons,
    text="Hallar Calculos",
    command=lambda: Boxes(tomografia,Prom_A.get(),Prom_R.get(),Desv_R.get(),Cota.get()),
    **button_style
).pack(
    fill=tk.BOTH,
    expand=True,
    pady=5,
    padx=10
)

tk.Button(
    frame_buttons,
    text="Marcar Calculos",
    command=lambda:obtener_coordenadas(),
    **button_style
).pack(
    fill=tk.BOTH,
    expand=True,
    pady=5,
    padx=10
)

tk.Button(
    frame_buttons,
    text="Clasificar Calculos",
    command=lambda: clasificacion(),
    **button_style
).pack(
    fill=tk.BOTH,
    expand=True,
    pady=5,
    padx=10
)

#Mantiene la aplicacion en constante actualizacion, permite interactuar con la interfaz
app.mainloop()
