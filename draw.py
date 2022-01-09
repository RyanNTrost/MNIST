import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.transforms import ToTensor

class ImageGenerator:
    def __init__(self,parent,posx,posy, model, *kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.model = model
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Done!",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)
        self.prediction = tk.StringVar(self.parent, value='Prediction:')
        self.prediction_label=tk.Label(self.parent, textvariable=self.prediction, font=("Arial", 25))
        self.prediction_label.place(x=300, y=50)

        self.image=Image.new("L",(200,200),(0))
        self.draw=ImageDraw.Draw(self.image)

    def save(self):
        filename = "data/custom/img.jpg"
        image = self.image.resize((28, 28), 1)
        image.save(filename)

    def predict(self):
        image = self.image.resize((28, 28), 1)
        image = ToTensor()(image)
        image = image.reshape(image.shape[0], -1)
        image = image.to('cuda')

        output = self.model(image)
        _, index = output.max(1)
        self.prediction.set(f'Prediction: {index.item()}')

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("L",(200,200),(0))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_oval(event.x, event.y, event.x, event.y, width=10, fill='black')
                self.draw.ellipse(((event.x - 10, event.y - 10),(event.x + 10,event.y + 10)), fill='white', width=10)

                self.predict()
        self.xold = event.x
        self.yold = event.y


if __name__ == "__main__":
    root = tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10, None)
    root.mainloop()