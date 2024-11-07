import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle
import os

# Load the pickled model
with open('digit_recognizer.pkl', 'rb') as file:
    model = pickle.load(file)

# Set DPI to match canvas size and buffer size more accurately
dpi = 100  # You can adjust this value as needed

# Create a square figure (640x640) with a specific DPI
fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=dpi)  # Square canvas 640x640 with the specified dpi
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
# ax.set_title('Draw a digit with your mouse (Live Recognition)')
ax.axis('off')

# Variables to store the drawing data
lines = []
current_line = []
last_input_image = None

# Function to recognize the digit while drawing
def recognize_digit_live():
    global last_input_image
    # Get the image from the canvas as RGBA and convert to grayscale
    fig.canvas.draw()
    X = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    X = X.reshape(int(height*1.5), int(width*1.5), 4)

    # Convert the RGBA image to grayscale
    img = Image.fromarray(X).convert('L')

    # Inverse the grayscale image by subtracting pixel values from 255
    img_inverted = Image.fromarray(255 - np.array(img))

    # Resize the image to 12x12 pixels (updated from 14x14)
    img_resized = img_inverted.resize((12, 12), Image.Resampling.LANCZOS)

    # Convert the resized image back to a numpy array and normalize the pixel values to [0, 1]
    X_resized = np.array(img_resized) / 255.0

    last_input_image = np.array(img_resized)

    # Flatten the image to a 1D array for the model
    X_flattened = X_resized.flatten()

    # Use the model's forward_pass method to recognize the digit
    probabilities  = model.forward_pass(X_flattened)
    predicted_digit = np.argmax(probabilities)  # The digit with the highest probability
    confidence = np.max(probabilities)  # The confidence (max probability)

    # Check if the confidence is above 50%
    if confidence >= 0.8:
        print(f"Recognized digit: {predicted_digit} with confidence {100*confidence:.2f}")
        # ax.set_title(f"Recognized digit: {predicted_digit} (Confidence: {confidence:.2f})")
    else:
        print(f"Recognized digit: Unknown (Confidence: {100*confidence:.2f})")
        # ax.set_title(f"Recognized digit: Unknown (Confidence: {confidence:.2f})")
    
    fig.canvas.draw()

def save_last_input_as_png():
    if last_input_image is not None:
        last_input_image.save("last_input.png")
        print("Saved last input image as 'last_input.png'")
    else:
        print("No input image to save.")

# Function to handle mouse clicks
def on_click(event):
    global current_line
    # Start a new line if the left mouse button is pressed
    if event.button == 1:  # Left-click
        current_line = [(event.xdata, event.ydata)]
        lines.append(current_line)
    elif event.button == 3:  # Right-click
        # Clear the drawing and reset the plot
        ax.clear()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        # ax.set_title('Draw a digit with your mouse (Live Recognition)')
        fig.canvas.draw()
# Function to handle mouse movement
def on_move(event):
    global current_line
    if event.button == 1 and event.xdata and event.ydata:
        current_line.append((event.xdata, event.ydata))
        line_xs, line_ys = zip(*current_line)
        ax.plot(line_xs, line_ys, color='black', linewidth=40)  # Adjust the linewidth here
        fig.canvas.draw()

        # Recognize the digit live while drawing
        recognize_digit_live()

def save_last_input_image():
    if last_input_image is not None:
        img = Image.fromarray(last_input_image)
        output_file = "last_input.png"
        img.save(output_file)
        print(f"Saved last input as {output_file}")
    else:
        print("No input image to save.")

def on_close(event):
    save_last_input_image()


# Function to handle mouse release
def on_release(event):
    pass  # Nothing to do here for now

# Connect the event handlers to the figure
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('motion_notify_event', on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('close_event', on_close)  # Handle window close event

# Display the interactive plot
plt.show()