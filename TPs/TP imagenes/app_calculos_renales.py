import tkinter as tk
from tkinter import filedialog, messagebox, DoubleVar
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import pandas as pd

#messagebox.showinfo(message="¡Hola, mundo!", title="Saludo")

def cargar_imagen():
    # Abrir el cuadro de diálogo para seleccionar el archivo
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
        print("No se seleccionó ninguna imagen.")
        return None

def plotear_imagen(imagen):
    ax.clear()
    plt.imshow(imagen, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    canvas.draw()

def seleccionar_y_mostrar():
    imagen = cargar_imagen()
    plotear_imagen(imagen)
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
    
def cargar_labels():
   # aca pongan el path a su carpeta de imagenes del drive
    train_labels_dir = 'C:/Users/Juan Bautista/.vscode/PSIB/TPs/TP imagenes/Dataset/train/labels'
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
def promedios_cajas(Areas, Ratios):
    prom_area = np.mean(Areas)
    prom_ratio = np.mean(Ratios)
    desv_area = np.std(Areas)
    desv_ratio = np.std(Ratios)
    return prom_area,desv_area, prom_ratio, desv_ratio
def apertura(imgbinaria):
  kernel = np.ones((9, 9), 'uint8')
  erode_img = cv2.erode(imgbinaria, kernel, iterations=1)

  kernel = np.ones((4, 4), 'uint8')
  close_img = cv2.dilate(erode_img, kernel, iterations=1)
  return close_img
def Boxes(img,Prom_A,Prom_R,Desv_R,Cota):
    print(Prom_A,Prom_R,Desv_R,Cota)
    ax.clear()
    img_cluster, centros=kmeans(img,3,10,0.9)
    imgbin=binarizar(img_cluster,centros)[1]
    img_Canny = cv2.Canny(imgbin, 200, 256)
    contours, _ = cv2.findContours(img_Canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copia = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (50+w<x+w<150 or 250+w<x+w<330):
            if (Prom_A*(1-Cota) < w*h < Prom_A*(1+Cota)) and (Prom_R - Desv_R < w/h < Prom_R + Desv_R and w< 50 and h<50):
                    print(f"Drawing rectangle at: x={x}, y={y}, w={w}, h={h}")
                    cv2.rectangle(copia, (x, y), (x+w, y+h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(copia, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    canvas.draw()


def onclick(event):
    global x, y
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        # Draw a marker at the clicked coordinate
        ax.plot(x, y, 'ro')  # 'ro' means red color, circle marker
        plt.draw()

def obtener_coordenadas():
    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

def region_growing():
    global tomografia, cid, x, y,region
    if 'x' not in globals() or 'y' not in globals():
        messagebox.showerror("Error", "Por favor, selecciona una coordenada primero.")
        return
    print(x, y)
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
    print("Region growing finalizado")
    ax.clear()
    ax.imshow(region, cmap='gray')
    ax.axis('off')
    canvas.draw()
    print("Region growing finalizado")
    return region
def clusters_y_bolainas():
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

def distancia(wc,hc,h,w):
  return np.sqrt((hc-h)**2+(wc-w)**2)

def clasificacion():
    global centersord, grado,centers,region
    obtener_coordenadas()
    region = region_growing()
    region_canny = cv2.Canny(region, 10, 256)
    contours, _ = cv2.findContours(region_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
    distancias=np.zeros(len(centers)) # Changed from centers[0] to centers
    for i in range(len(centers)): # Changed from centers[0] to centers
        distancias[i]=distancia(centersord[i][0],centersord[i][1],h,w) # Changed from centers[0][i][0] to centers[i][0] and centers[0][i][1] to centers[i][1]
        min(distancias)
    for i in range(len(distancias)):
        if distancias[i]==min(distancias):
            grado = str(i+1)
            print(f"El cálculo es de Grado {i+1}")
            messagebox.showinfo("Clasificación", f"El cálculo es de Grado {i+1}")
tomografia = None
app = tk.Tk()
grado = tk.StringVar(app,value = "Sin calcular")
cargar_labels()
A,R,C,D = analisis_cajas(cargar_labels())
PrA, DesA, PrR, DesR = promedios_cajas(A,R)
palabra = tk.StringVar(app)
entrada = tk.StringVar(app)
Prom_A = tk.DoubleVar( value=PrA)
Prom_R = tk.DoubleVar( value=PrR)
Desv_R = tk.DoubleVar( value=DesR)
Cota = tk.DoubleVar( value=2)
print(Prom_A.get(),Prom_R.get(),Desv_R.get(),Cota.get())
#####

widths = [dim[0] for dim in D]
heights = [dim[1] for dim in D]
clusters_y_bolainas()
#Dimensiones: Ancho x Altura
app.geometry("1000x1000")
#Cambio el color del fondo
app.configure(bg="black")
#Titulo de la ventana
app.title("Detector de calculos renales")

#prueba de grafico
fig, ax = plt.subplots()
plt.imshow(np.zeros((320,391)), cmap='gray', vmin=0, vmax=255)
ax.axis('off')
canvas = FigureCanvasTkAgg(fig, master=app)
canvas.get_tk_widget().pack()
herramienas = NavigationToolbar2Tk(canvas, app, pack_toolbar=False)
herramienas.update()
herramienas.pack(anchor='nw',fill='both')

#Boton para cargar la imagen

tk.Button(
    app,
    text="Cargar imagen",
    font=("Arial", 12),
    bg = "#00a0f0",
    fg = "black",
    command=seleccionar_y_mostrar,
    relief="solid",
).pack(
    fill=tk.BOTH,
    expand=True,
)
tk.Button(
    app,
    text="Hallar Calculos",
    font=("Arial", 12),
    bg = "green",
    fg = "black",
    command=lambda: Boxes(tomografia,Prom_A.get(),Prom_R.get(),Desv_R.get(),Cota.get()),
    relief="flat",
).pack(
    fill=tk.BOTH,
    expand=True,
)
tk.Button(
    app,
    text="Marcar Calculos",
    font=("Arial", 12),
    bg = "magenta",
    fg = "black",
    command=lambda:obtener_coordenadas(),
    relief="flat",
).pack(
    fill=tk.BOTH,
    expand=True,
)
tk.Button(
    app,
    text="Clasificar Calculos",
    font=("Arial", 12),
    bg = "yellow",
    fg = "black",
    command=lambda: clasificacion(),
    relief="flat",
).pack(
    fill=tk.BOTH,
    expand=True,
)
#Mantiene la aplicacion en constante actualizacion, permite interactuar con la interfaz
app.mainloop()
