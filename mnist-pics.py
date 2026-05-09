#mnist_pics.py


from PIL import Image, ImageDraw
import os

# Create dataset folders
for i in range(10):
    os.makedirs(f"data/{i}", exist_ok=True)

class MNISTDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Drawing App")

        self.canvas_size = 280  # large canvas for drawing
        self.image_size = 28    # MNIST size

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        for i in range(10):
            btn = tk.Button(self.button_frame, text=str(i), command=lambda i=i: self.save(i))
            btn.grid(row=0, column=i)

        self.clear_btn = tk.Button(root, text="Clear", command=self.clear)
        self.clear_btn.pack()

        # PIL image for saving
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw_image.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw_image = ImageDraw.Draw(self.image)

    def save(self, label):
        # Resize to MNIST size
        img = self.image.resize((self.image_size, self.image_size))

        # Save image
        count = len(os.listdir(f"data/{label}"))
        filename = f"data/{label}/{count}.png"
        img.save(filename)

        print(f"Saved {filename}")
        self.clear()

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawer(root)
    root.mainloop()