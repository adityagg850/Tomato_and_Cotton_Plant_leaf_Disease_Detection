from tkinter import *
import os
root = Tk()
img_size = 48
batch_size = 64
def training():
    os.system("py cnn.py")
def plotacc():
    os.system("py cnnacc.py")
def plotacc1():
    os.system("py cnnacctom.py")
def function6():
    root.destroy()
def appopen():
    os.system("py webapp.py")
def appopen1():
    os.system("py test.py")
root.configure(background="white")
root.title("Leaf Disease Detection")
# creating a text label
Label(root, text="Leaf Disease Detection", font=("times new roman", 20), fg="white", bg="#1A3C40",
      height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Model Building", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=training).grid(
    row=1, columnspan=2, sticky=N + E + W + S,padx=75, pady=15)
Button(root, text="Model Accuracy(Cotton Disease)", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=plotacc).grid(
    row=2, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Model Accuracy(Tomato Disease)", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=plotacc1).grid(
    row=3, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
# creating second button
Button(root, text="Tomato Disease Detection Web App", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=appopen).grid(
    row=4, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Cotton Disease Detection Web App", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=appopen1).grid(
    row=5, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Exit", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=function6).grid(
    row=6, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()
