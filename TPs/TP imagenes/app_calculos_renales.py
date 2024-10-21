import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
app = tk.Tk()
palabra = tk.StringVar(app)
entrada = tk.StringVar(app)
def cargar_imagen():
    # Abrir el cuadro de diálogo para seleccionar el archivo
    file_path = filedialog.askopenfilename(title="Selecciona una imagen", 
                                        filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff")])

    if file_path:
        # Abrir y mostrar la imagen seleccionada
        img = Image.open(file_path)
        img.show()  # Muestra la imagen en la aplicación predeterminada
        imagen = cv2.imread(file_path, 0)
    else:
        print("No se seleccionó ninguna imagen.")
    


#Dimensiones: Ancho x Altura
app.geometry("600x600")
#Cambio el color del fondo
app.configure(bg="black")
#Titulo de la ventana
app.title("Detector de calculos renales")
tk.Button(
    app,
    text="Cargar imagen",
    font=("Arial", 12),
    bg = "#00a0f0",
    fg = "black",
    command=cargar_imagen,
    relief="flat",
).pack(
    fill=tk.BOTH,
    expand=True,
)

tk.Label(
    app,
    text="Coordenadas",
    font=("Arial", 12),
    bg = "black",
    fg = "white",
    justify="center"
).pack(
    fill=tk.BOTH,
    expand=True,
)

tk.Entry(
    app,
    font=("Arial", 12),
    bg = "white",
    fg = "black",
    justify="center",
    textvariable=entrada,
).pack(
    fill=tk.BOTH,
    expand=True,
)
#Mantiene la aplicacion en constante actualizacion, permite interactuar con la interfaz
app.mainloop()
