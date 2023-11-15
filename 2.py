import numpy as np
from PIL import Image, ImageTk
from feature_extractor import FeatureExtractor
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path


class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Search App")

        self.fe = FeatureExtractor()
        self.features = []
        self.img_paths = []
        self.query_img = None
        self.query_img_tk = None
        self.query_features = None

        for feature_path in Path("./static/feature").glob("*.npy"):
            self.features.append(np.load(feature_path))
            self.img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
        self.features = np.array(self.features)

        self.create_widgets()

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)
        self.exit_button.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.query_img = Image.open(file_path)
            self.query_img = self.query_img.resize((224, 224))  # Resize the image
            self.query_img_tk = ImageTk.PhotoImage(self.query_img)
            self.query_features = self.fe.extract(self.query_img)

            if not hasattr(self, "uploaded_label"):
                self.uploaded_label = tk.Label(self.root, text="Uploaded Image:")
                self.uploaded_label.pack()

            if hasattr(self, "img_label"):
                self.img_label.configure(image=self.query_img_tk)
                self.img_label.image = self.query_img_tk
            else:
                self.img_label = tk.Label(self.root, image=self.query_img_tk)
                self.img_label.image = self.query_img_tk
                self.img_label.pack()

            if not hasattr(self, "similarity_button"):
                self.similarity_button = tk.Button(self.root, text="Similarity", command=self.show_similarity)
                self.similarity_button.pack()

            if not hasattr(self, "relevant_button"):
                self.relevant_button = tk.Button(self.root, text="Relevant Images", command=self.show_relevant_images)
                self.relevant_button.pack()

    def show_uploaded_image(self):
        uploaded_label = tk.Label(self.root, text="Uploaded Image:")
        uploaded_label.pack()

        img_label = tk.Label(self.root, image=self.query_img_tk)
        img_label.image = self.query_img_tk
        img_label.pack()

        similarity_button = tk.Button(self.root, text="Similarity", command=self.show_similarity)
        similarity_button.pack()

        relevant_button = tk.Button(self.root, text="Relevant Images", command=self.show_relevant_images)
        relevant_button.pack()

    def show_similarity(self):
        if self.query_features is not None and len(self.query_features) > 0:
            dists = np.linalg.norm(self.features - self.query_features, axis=1)
            ids = np.argsort(dists)[:30]
            scores = [(dists[id],) for id in ids]
            self.display_scores(scores)

            # accuracy = self.calculate_accuracy()
            # self.plot_accuracy_graph()
        else:
            messagebox.showerror("Error", "No query image to calculate similarity for.")


    def show_relevant_images(self):
        if self.query_features is not None and len(self.query_features) > 0:
            relevant_imgs = []
            dists = np.linalg.norm(self.features - self.query_features, axis=1)
            ids = np.argsort(dists)[:10]
            for id in ids:
                img_path = self.img_paths[id]
                img = Image.open(img_path)
                img_tk = ImageTk.PhotoImage(img)
                relevant_imgs.append(img_tk)

            self.display_relevant_images(relevant_imgs)
        else:
            messagebox.showerror("Error", "No query image to find relevant images for.")

    def display_scores(self, scores):
        score_window = tk.Toplevel(self.root)
        score_window.title("Similarity Scores")

        for score in scores:
            score_label = tk.Label(score_window, text=f"Score: {score[0]:.2f}")
            score_label.pack()

        exit_button = tk.Button(score_window, text="Exit", command=score_window.destroy)
        exit_button.pack()


    def display_relevant_images(self, relevant_imgs):
        relevant_window = tk.Toplevel(self.root)
        relevant_window.title("Relevant Images")

        canvas = tk.Canvas(relevant_window)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(relevant_window, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        image_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=image_frame, anchor="nw")

        num_columns = 4
        for idx, img_tk in enumerate(relevant_imgs, start=1):
            row = (idx - 1) // num_columns
            col = (idx - 1) % num_columns

            img_label = tk.Label(image_frame, image=img_tk)
            img_label.image = img_tk
            img_label.grid(row=row, column=col, padx=5, pady=5, sticky="nw")

        exit_button = tk.Button(relevant_window, text="Exit", command=relevant_window.destroy)
        exit_button.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()
