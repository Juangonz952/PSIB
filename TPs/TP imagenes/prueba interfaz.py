import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV
image_path = 'C:/Users/Juan Bautista/.vscode/PSIB/Texturas/drive-download-20241009T222141Z-001/img17.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

# Create a figure and axis to display the image
fig, ax = plt.subplots()
ax.imshow(image)

# List to store coordinates
coords = []

# Function to capture click events
def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        coords.append((x, y))
        # Draw a marker at the clicked coordinate
        ax.plot(x, y, 'ro')  # 'ro' means red color, circle marker
        plt.draw()

# Connect the click event to the function
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Display the image
plt.show()

# Print the coordinates after the window is closed
print("Coordinates:", coords)